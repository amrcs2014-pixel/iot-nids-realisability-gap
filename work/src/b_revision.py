"""Paper B - revised analyses on the executed knob library (Oct 2026). Writes results/b_revision.json.

One consistent, cross-fitted protocol (every flow is scored by a model that never trained on its run or config):
  R1  flow shape per family and the knob x family implementation matrix (which knobs each attack reads)
  R2  four detectors (XGBoost, random forest, MLP, benign-only isolation forest) x two training regimes:
        natural  - attacks from the 12 natural-configuration runs per family only (+ benign)
        knob     - knob-augmented: natural + Sobol configurations, configs held out by grouped CV
      threshold = 99th percentile of out-of-fold benign scores (1% FPR); clean recall on held-out natural runs;
      realised evasion of FUNCTIONAL Sobol runs: random-knob, surrogate-transfer, query-feedback (flow level)
      and the number of functional runs that raise no alert at all (run level)
  R3  feature-space arm per detector x regime: line search and HopSkipJump (in standardised feature space),
      ASR, median L2 in raw and standardised units, feasibility labels, displacement shares by feature group
  R4  ablations of the dominant features (drop dst2src_bytes + bidirectional_min_ps; drop all victim-side
      features; drop all byte/size features; log2-quantised values)
  R5  controllability of dst2src_bytes / bidirectional_min_ps: knob correlations and overlap with benign
  R6  GothamDataset2025 in both directions: Gotham-trained detector on the twin knob library; twin-trained
      detector re-calibrated to 1% FPR on held-out REAL benign
  R7  ElShehaby-Matrawy perturb-ability score (five sub-scores) for every feature vs executed reachability
"""
import collections, faulthandler, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
if os.environ.get("STACKDUMP"):
    faulthandler.dump_traceback_later(int(os.environ["STACKDUMP"]), repeat=True)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_detect import feature_cols
from b_attack import invariants_ok, CTRL_PREFIX
try:
    from art.estimators.classification import BlackBoxClassifier
    from art.attacks.evasion import HopSkipJump
    _ART = True
except Exception:
    _ART = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", os.environ.get("BDATA", "btwin")); RES = os.path.join(ROOT, "results")
FAMS = ["netscan", "svcbrute", "synflood", "coapamp", "httpbeacon"]
NAT = {"rate_pps": 200.0, "ipd_jitter": 0.0, "pad_bytes": 0, "pkts_per_conn": 1, "parallel": 1, "srcport_mode": 2,
       "ttl": 64, "tcp_window": 29200.0, "frag_mtu": 0, "netem_delay": 0.0, "netem_loss": 0.0, "idle_insert": 0.0, "cover": 0.0}
# knobs read by each attack implementation (b_hosts.py); netem delay/loss are applied by the harness to every family;
# `cover` is read by no attack (it was sampled but never changed any traffic)
IMPL = {"netscan": {"rate_pps", "ipd_jitter", "pad_bytes", "srcport_mode", "ttl", "tcp_window", "idle_insert"},
        "svcbrute": {"rate_pps", "ipd_jitter", "pad_bytes", "pkts_per_conn", "idle_insert"},
        "synflood": {"rate_pps", "ipd_jitter", "pad_bytes", "parallel", "srcport_mode", "ttl", "tcp_window", "frag_mtu", "idle_insert"},
        "coapamp": {"rate_pps", "ipd_jitter", "srcport_mode", "ttl", "idle_insert"},
        "httpbeacon": {"rate_pps", "ipd_jitter", "pad_bytes"}}
for _f in IMPL:
    IMPL[_f] |= {"netem_delay", "netem_loss"}
R = np.random.default_rng(3)
T0 = time.time()
QUICK = os.environ.get("QUICK") == "1"
NJOBS = int(os.environ.get("NJOBS", 8))      # use NJOBS=1 (and low priority) while the twin is generating
N_FS = 12 if QUICK else 40          # feature-space attack samples per family
N_HSJA = 4 if QUICK else 10
# Hard query budget per HopSkipJump run (exhausted -> counted as a failure). A successful run needs ~1,450 queries;
# where the detector is locally flat (trees/benign-only), ART's gradient estimate degenerates and its step-size
# loop queries one point at a time until the cap, each query a slow single-row ensemble predict -- so we also cap
# the wall-clock time spent on HopSkipJump per family (the line-search and masked ASR are the primary metrics;
# HopSkipJump is a corroborating check, reported over however many samples complete within the budget).
HSJA_QUERY_CAP = 1200
HSJA_FAM_BUDGET = float(os.environ.get("HSJA_FAM_BUDGET", 45))


class QueryBudget(Exception):
    pass


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ data
raw = pd.read_parquet(os.path.join(DATA, "flows.parquet"))
cal = pd.read_parquet(os.path.join(DATA, "benign_cal_flows.parquet"))
att = raw[raw["label"] == 1].copy()
for c in raw.columns:
    if c not in cal.columns:
        cal[c] = np.nan
df = pd.concat([att, cal.assign(label=0)[raw.columns]], ignore_index=True)
FEATS = list(feature_cols(df))
df[FEATS] = df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
is_nat = np.ones(len(df), bool)
for k, v in NAT.items():
    is_nat &= np.isclose(df["k_" + k].astype(float).fillna(-999).values, float(v))
df["natural"] = (df["label"] == 1).values & is_nat
df["cfg_group"] = np.where(df["label"] == 1, np.where(df["natural"], "nat_" + df["family"], df["run_id"]), "ben_" + df["run_id"].astype(str))
if QUICK:
    df = df.loc[df.groupby("run_id").head(25).index].reset_index(drop=True)
X = df[FEATS].values.astype(float); y = df["label"].values; fam = df["family"].values
func = df["func_ok"].fillna(1).values.astype(int); natural = df["natural"].values; runs = df["run_id"].values
BEN = np.where(y == 0)[0]
MU, SD = X[BEN].mean(0), X[BEN].std(0); SD[SD < 1e-9] = 1.0
out = {"n_flows": int(len(df)), "n_features": len(FEATS), "features": FEATS,
       "n_benign_flows": int(len(BEN)), "n_benign_runs": int(df.loc[BEN, "run_id"].nunique()),
       "n_attack_flows": int((y == 1).sum()), "n_natural_runs": int(df.loc[natural, "run_id"].nunique())}
log("data", {k: v for k, v in out.items() if k != "features"})

# ------------------------------------------------------------------ R1 flow shape + implementation matrix
shape = {}
for f in FAMS:
    a = df[(y == 1) & (fam == f)]
    shape[f] = {"flows": int(len(a)), "runs": int(a["run_id"].nunique()),
                "flows_per_run_median": float(a.groupby("run_id").size().median()),
                "pkts_per_flow_median": float(a["bidirectional_packets"].median()),
                "fwd_pkts_median": float(a["src2dst_packets"].median()), "bwd_pkts_median": float(a["dst2src_packets"].median()),
                "duration_ms_median": float(a["bidirectional_duration_ms"].median()),
                "func_natural": [int(a[a.natural].groupby("run_id").func_ok.first().sum()), int(a[a.natural].run_id.nunique())],
                "func_sobol": [int(a[~a.natural].groupby("run_id").func_ok.first().sum()), int(a[~a.natural].run_id.nunique())]}
out["R1_shape"] = shape
out["R1_impl"] = {f: sorted(v) for f, v in IMPL.items()}
coap = df[(y == 1) & (fam == "coapamp")].groupby("run_id").agg(req=("src2dst_bytes", "sum"), resp=("dst2src_bytes", "sum"))
amp = coap.resp / coap.req.clip(lower=1)
out["R1_coap_amplification"] = {"median": float(amp.median()), "min": float(amp.min()), "max": float(amp.max()), "runs": int(len(coap))}
log("R1", {f: (v["func_natural"], v["func_sobol"], v["pkts_per_flow_median"]) for f, v in shape.items()}, "coap amp", out["R1_coap_amplification"])


# ------------------------------------------------------------------ models
def make(kind):
    if kind == "xgb":
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                                 eval_metric="logloss", tree_method="hist", n_jobs=NJOBS, random_state=0)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=NJOBS, random_state=0, class_weight="balanced")
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0)
    raise ValueError(kind)


class Model:
    """Uniform wrapper: score(X) = attack score (higher = more attack-like)."""
    def __init__(self, kind, cols=None, quant=False):
        self.kind, self.cols, self.quant = kind, cols, quant

    def _prep(self, Xa):
        Z = Xa if self.cols is None else Xa[:, self.cols]
        if self.quant:
            Z = np.sign(Z) * np.floor(np.log2(np.abs(Z) + 1))
        return Z

    def fit(self, Xa, ya):
        Z = self._prep(Xa)
        if self.kind == "iso":
            self.sc = StandardScaler().fit(Z[ya == 0])
            self.m = IsolationForest(n_estimators=200, random_state=0).fit(self.sc.transform(Z[ya == 0]))
        elif self.kind == "mlp":
            self.sc = StandardScaler().fit(Z); self.m = make("mlp").fit(self.sc.transform(Z), ya)
        else:
            w = np.where(ya == 1, 0.5 / max(1e-9, ya.mean()), 0.5 / max(1e-9, 1 - ya.mean()))
            self.m = make(self.kind).fit(Z, ya, sample_weight=w)
        return self

    def score(self, Xa):
        Z = self._prep(np.atleast_2d(Xa))
        if self.kind == "rf":
            self.m.n_jobs = NJOBS if len(Z) > 4000 else 1
        if self.kind == "iso":
            return -self.m.score_samples(self.sc.transform(Z))
        if self.kind == "mlp":
            return self.m.predict_proba(self.sc.transform(Z))[:, 1]
        return self.m.predict_proba(Z)[:, 1]


class Ensemble:
    def __init__(self, ms):
        self.ms = ms; self.kind = ms[0].kind

    def score(self, Xa):
        return np.mean([m.score(Xa) for m in self.ms], axis=0)


ATT = np.where(y == 1)[0]
ben_folds = list(GroupKFold(5).split(BEN, groups=df.loc[BEN, "run_id"].values))
att_folds = list(GroupKFold(5).split(ATT, groups=df.loc[ATT, "cfg_group"].values))
NATIDX = np.where(natural)[0]
nat_folds = list(GroupKFold(5).split(NATIDX, groups=runs[NATIDX]))
SOB = np.where((y == 1) & ~natural)[0]


def crossfit(kind, regime, cols=None, quant=False):
    """OOF attack score for every flow, the scorer index per flow (5 = ensemble of the fold models) and the scorers.
    knob: grouped 5-fold over all attack configs + benign runs.
    natural: train on natural runs (+benign) of 4 folds, score the held-out natural runs and benign runs;
             every Sobol flow is scored by the average of the 5 fold models (none of them saw any Sobol config)."""
    s = np.full(len(df), np.nan); who = np.full(len(df), -1); models = []
    for k in range(5):
        btr, bte = BEN[ben_folds[k][0]], BEN[ben_folds[k][1]]
        if regime == "knob":
            tr = np.r_[ATT[att_folds[k][0]], btr]; te = np.r_[ATT[att_folds[k][1]], bte]
        else:
            tr = np.r_[NATIDX[nat_folds[k][0]], btr]; te = np.r_[NATIDX[nat_folds[k][1]], bte]
        m = Model(kind, cols, quant).fit(X[tr], y[tr])
        s[te] = m.score(X[te]); who[te] = k; models.append(m)
    if regime == "natural":
        models.append(Ensemble(models[:5]))
        s[SOB] = models[5].score(X[SOB]); who[SOB] = 5
    return s, who, models


def realised(s, thr, prob=True):
    """Realised evasion over FUNCTIONAL Sobol runs of each family (flow level + run level)."""
    res = {}
    for f in FAMS:
        idx = SOB[(fam[SOB] == f) & (func[SOB] == 1)]
        if len(idx) == 0:
            res[f] = {"n_func_flows": 0, "n_func_runs": 0}; continue
        sc = s[idx]; ev = sc <= thr
        # surrogate transfer: the attacker's own RF trained on half of the family's Sobol configs (+ benign) picks
        # the 10% most benign-looking functional flows of the other half; evaluated on the real detector
        cfgs = np.array(sorted(set(runs[idx]))); R.shuffle(cfgs); halves = [set(cfgs[: len(cfgs) // 2]), set(cfgs[len(cfgs) // 2:])]
        tr_frac = []
        fam_all = SOB[fam[SOB] == f]
        for hA, hB in ((0, 1), (1, 0)):
            trn = np.r_[fam_all[np.isin(runs[fam_all], list(halves[hA]))], BEN]
            cand = idx[np.isin(runs[idx], list(halves[hB]))]
            if len(set(y[trn])) < 2 or not len(cand):
                continue
            sur = RandomForestClassifier(n_estimators=100, n_jobs=NJOBS, random_state=1, class_weight="balanced").fit(X[trn], y[trn])
            pick = cand[np.argsort(sur.predict_proba(X[cand])[:, 1])[: max(1, len(cand) // 10)]]
            tr_frac.append(float(np.mean(s[pick] <= thr)))
        q = idx[np.argsort(sc)[: max(1, len(idx) // 10)]]
        per_run = pd.Series(sc).groupby(runs[idx]).max()
        mn = float(sc.min())
        if prob and 0 < thr < 1:
            lo = lambda p: np.log(max(p, 1e-12) / max(1 - p, 1e-12))
            margin = float(lo(mn) - lo(thr))
        else:
            margin = float(mn - thr)
        res[f] = {"n_func_flows": int(len(idx)), "n_func_runs": int(len(per_run)),
                  "r_rand": float(ev.mean()), "r_transfer": float(np.mean(tr_frac)) if tr_frac else None,
                  "r_query": float(np.mean(s[q] <= thr)), "runs_no_alert": int((per_run <= thr).sum()),
                  "closest_margin": margin}
    return res


def clean_recall(s, thr):
    return {f: (float(np.mean(s[NATIDX[fam[NATIDX] == f]] > thr)) if (fam[NATIDX] == f).any() else None) for f in FAMS}


def sobol_recall(s, thr):
    return {f: float(np.mean(s[SOB[fam[SOB] == f]] > thr)) for f in FAMS}


# ------------------------------------------------------------------ R2 detectors x regimes
R2 = {}; OOF = {}; SC = {}
for kind in ("xgb", "rf", "mlp", "iso"):
    for regime in (("knob", "natural") if kind != "iso" else ("benign_only",)):
        s, who, models = crossfit(kind, "knob" if regime == "benign_only" else regime)
        thr = float(np.quantile(s[BEN], 0.99))
        ok = ~np.isnan(s)
        key = f"{kind}:{regime}"
        ben_run_alarm = float(pd.Series(s[BEN] > thr).groupby(runs[BEN]).any().mean())
        R2[key] = {"auc_oof": float(roc_auc_score(y[ok], s[ok])), "thr": thr, "fpr_oof": float(np.mean(s[BEN] > thr)),
                   "benign_run_alarm_rate": ben_run_alarm,
                   "clean_recall_natural": clean_recall(s, thr), "sobol_recall_all": sobol_recall(s, thr),
                   "realised": realised(s, thr, prob=(kind != "iso"))}
        OOF[key] = (s, thr); SC[key] = (models, who)
        r = R2[key]["realised"]
        log("R2", key, "AUC", round(R2[key]["auc_oof"], 4), "clean", {f: round(v, 2) for f, v in R2[key]["clean_recall_natural"].items() if v is not None},
            "r_rand", {f: round(v.get("r_rand", np.nan), 3) for f, v in r.items()}, "runs_no_alert", {f: (v.get("runs_no_alert"), v.get("n_func_runs")) for f, v in r.items()})
out["R2_detectors"] = R2

# ------------------------------------------------------------------ R3 feature-space arm
MASK_CTRL = np.array([c.startswith(CTRL_PREFIX) for c in FEATS])
GRP = np.array(["victim" if c.startswith("dst2src_") else "attacker" if c.startswith("src2dst_") else "whole" for c in FEATS])


def batch_boundary(scorer, thr, X0, T, steps=24):
    """Vectorised decision-based line search from each row of X0 toward the matching row of T (a benign
    reference, or X0 with only the allowed features replaced). Returns the boundary points and success flags."""
    ok_end = scorer.score(T) <= thr
    lo = np.zeros(len(X0)); hi = np.ones(len(X0)); D = T - X0
    for _ in range(steps):
        mid = (lo + hi) / 2
        ev = scorer.score(X0 + mid[:, None] * D) <= thr
        hi = np.where(ev, mid, hi); lo = np.where(ev, lo, mid)
    Xa = X0 + hi[:, None] * D
    return Xa, ok_end & (scorer.score(Xa) <= thr)


def fs_attack(key, n_ref=6):
    s, thr = OOF[key]; models, who = SC[key]
    pool_all = BEN[s[BEN] <= thr]
    res = {}
    for f in FAMS:
        cand = SOB[(fam[SOB] == f) & (func[SOB] == 1) & (s[SOB] > thr)]
        if len(cand) < 3:
            res[f] = {"n": int(len(cand))}; continue
        samp = R.choice(cand, min(N_FS, len(cand)), replace=False)
        best = {}; msk = {}; t_f = time.time()
        for g in sorted(set(who[samp])):
            sg = samp[who[samp] == g]; sc_ = models[g]
            refs = np.array([R.choice(pool_all, n_ref, replace=len(pool_all) < n_ref) for _ in sg])   # (n, n_ref)
            X0 = np.repeat(X[sg], n_ref, axis=0); Tu = X[refs.ravel()]
            Tm = X0.copy(); Tm[:, MASK_CTRL] = Tu[:, MASK_CTRL]
            Xa, ok = batch_boundary(sc_, thr, np.r_[X0, X0], np.r_[Tu, Tm])
            nu = len(X0)
            for j, i in enumerate(sg):
                rows = np.arange(j * n_ref, (j + 1) * n_ref)
                ou = rows[ok[rows]]
                if len(ou):
                    d = np.sum(((Xa[ou] - X[i]) / SD) ** 2, axis=1)
                    best[i] = Xa[ou[np.argmin(d)]]
                msk[i] = bool(ok[nu + rows].any())
        l2r = []; l2z = []; feas = collections.Counter(); disp = np.zeros(len(FEATS))
        for i, xa in best.items():
            l2r.append(float(np.linalg.norm(xa - X[i]))); l2z.append(float(np.linalg.norm((xa - X[i]) / SD)))
            moved = np.abs(xa - X[i]) > 1e-3 * (np.abs(X[i]) + 1)
            if not invariants_ok(xa, FEATS):
                feas["infeasible"] += 1
            elif moved[~MASK_CTRL].any():
                feas["needs_noncontrollable"] += 1
            else:
                feas["controllable_only"] += 1
            disp += np.abs(xa - X[i]) / SD
        share = {}
        if best:
            for g_ in ("attacker", "whole", "victim"):
                share[g_] = float(disp[GRP == g_].sum() / max(1e-12, disp.sum()))
        # HopSkipJump in standardised space, initialised from a benign reference
        hs_ok = []; hs_l2r = []; hs_l2z = []; t_hs = time.time()
        n_budget = 0
        if _ART:
            for i in samp[:N_HSJA]:
                if time.time() - t_hs > HSJA_FAM_BUDGET:       # a flat-boundary detector can make every HSJA call
                    break                                       # burn its full query cap; cap the family's total time
                sc_ = models[who[i]]
                used = [0]

                def oracle(Z, sc_=sc_, used=used):
                    Z = np.asarray(Z); used[0] += len(Z)
                    if used[0] > HSJA_QUERY_CAP:
                        raise QueryBudget()
                    return np.eye(2)[(sc_.score(Z * SD + MU) > thr).astype(int)]
                try:
                    bb = BlackBoxClassifier(oracle, input_shape=(len(FEATS),), nb_classes=2)
                    z0 = ((X[i] - MU) / SD)[None, :]; zi = ((X[R.choice(pool_all)] - MU) / SD)[None, :]
                    t_c = time.time()
                    if os.environ.get("HSJA_VERBOSE"):
                        log(f"      hsja start {key} {f} i={i} |z0|max={np.abs(z0).max():.3g} |zi|max={np.abs(zi).max():.3g}")
                    za = HopSkipJump(classifier=bb, targeted=False, norm=2, max_iter=20, max_eval=200, init_eval=20,
                                     verbose=False).generate(x=z0, x_adv_init=zi)
                    if os.environ.get("HSJA_VERBOSE"):
                        log(f"      hsja done {time.time() - t_c:.1f}s queries={used[0]}")
                    ok_ = bool(sc_.score(za[0] * SD + MU)[0] <= thr)
                    hs_ok.append(ok_)
                    if ok_:
                        hs_l2z.append(float(np.linalg.norm(za[0] - z0[0]))); hs_l2r.append(float(np.linalg.norm((za[0] - z0[0]) * SD)))
                except QueryBudget:
                    hs_ok.append(False); n_budget += 1          # budget exhausted counts as a failed attack
                except Exception as e:
                    log("hsja err", key, f, repr(e)[:80])
        n = len(samp); succ = len(best)
        log(f"   fs {key} {f}: line+mask {t_hs - t_f:.1f}s, hsja {time.time() - t_hs:.1f}s")
        res[f] = {"n": int(n), "asr_line": succ / n, "asr_masked": float(np.mean(list(msk.values()))),
                  "asr_hsja": float(np.mean(hs_ok)) if hs_ok else None, "n_hsja": len(hs_ok), "n_hsja_budget_exhausted": n_budget,
                  "l2_raw_line_med": float(np.median(l2r)) if l2r else None, "l2_z_line_med": float(np.median(l2z)) if l2z else None,
                  "l2_raw_hsja_med": float(np.median(hs_l2r)) if hs_l2r else None, "l2_z_hsja_med": float(np.median(hs_l2z)) if hs_l2z else None,
                  "feasibility": {k: feas[k] / max(1, succ) for k in ("infeasible", "needs_noncontrollable", "controllable_only")},
                  "displacement_share": share}
    return res


R3 = {}
FS_KEYS = [k for k in os.environ.get("FS_KEYS", ",".join(OOF)).split(",") if k in OOF]   # smoke tests may restrict
for key in FS_KEYS:
    R3[key] = fs_attack(key)
    log("R3", key, {f: (round(v.get("asr_line", np.nan), 2), round(v.get("asr_masked", np.nan), 2), v.get("asr_hsja"),
                        v.get("feasibility", {}).get("controllable_only")) for f, v in R3[key].items()})
out["R3_feature_space"] = R3

# ------------------------------------------------------------------ R4 ablations (XGBoost, both regimes)
ABL = {"drop_top2": [c for c in FEATS if c not in ("dst2src_bytes", "bidirectional_min_ps")],
       "drop_victim_side": [c for c in FEATS if not c.startswith("dst2src_")],
       "no_size_features": [c for c in FEATS if not (c.endswith("_bytes") or c.endswith("_ps"))],
       "log2_quantised": None}
R4 = {}
for regime in ("knob", "natural"):
    for name, cols in ABL.items():
        ci = None if cols is None else [FEATS.index(c) for c in cols]
        s, who, models = crossfit("xgb", regime, cols=ci, quant=(name == "log2_quantised"))
        thr = float(np.quantile(s[BEN], 0.99))
        imp = models[0].m.feature_importances_
        names = FEATS if ci is None else [FEATS[i] for i in ci]
        top = sorted(zip(names, imp), key=lambda t: -t[1])[:5]
        ok = ~np.isnan(s)
        key = f"xgb:{regime}:{name}"
        R4[key] = {"n_features": len(names), "auc_oof": float(roc_auc_score(y[ok], s[ok])),
                   "clean_recall_natural": clean_recall(s, thr), "sobol_recall_all": sobol_recall(s, thr),
                   "realised": realised(s, thr), "top_features": [(n_, float(v)) for n_, v in top],
                   "top3_gain_share": float(sum(v for _, v in top[:3]) / max(1e-12, imp.sum()))}
        log("R4", key, "AUC", round(R4[key]["auc_oof"], 4), "clean", {f: round(v, 2) for f, v in R4[key]["clean_recall_natural"].items()},
            "r_rand", {f: round(v.get("r_rand", np.nan), 3) for f, v in R4[key]["realised"].items()})
out["R4_ablation"] = R4
for regime in ("knob", "natural"):
    imp = SC[f"xgb:{regime}"][0][0].m.feature_importances_
    out[f"R4_full_top_features_{regime}"] = [(FEATS[i], float(imp[i])) for i in np.argsort(imp)[::-1][:5]]
    out[f"R4_full_top3_gain_share_{regime}"] = float(np.sort(imp)[::-1][:3].sum() / imp.sum())

# ------------------------------------------------------------------ R5 controllability of the dominant features
R5 = {}
FOC = ("dst2src_bytes", "bidirectional_min_ps", "src2dst_bytes")
bq = {c: (float(np.quantile(X[BEN, FEATS.index(c)], 0.01)), float(np.quantile(X[BEN, FEATS.index(c)], 0.99))) for c in FOC}
for f in FAMS:
    idx = SOB[(fam[SOB] == f) & (func[SOB] == 1)]
    if not len(idx):
        continue
    row = {}
    for c in FOC:
        v = X[idx, FEATS.index(c)]; lo, hi = bq[c]; rho = {}
        for k in IMPL[f]:
            kv = df[f"k_{k}"].values[idx].astype(float)
            if np.std(kv) > 0 and np.std(v) > 0:
                rho[k] = float(spearmanr(kv, v).correlation)
        row[c] = {"attack_median": float(np.median(v)), "attack_p05": float(np.percentile(v, 5)), "attack_p95": float(np.percentile(v, 95)),
                  "benign_p01": lo, "benign_p99": hi, "frac_in_benign_range": float(np.mean((v >= lo) & (v <= hi))),
                  "knob_spearman": dict(sorted(rho.items(), key=lambda t: -abs(t[1]))[:4])}
    R5[f] = row
out["R5_controllability"] = R5
out["R5_benign_quantiles"] = bq
log("R5", {f: {c: round(v["frac_in_benign_range"], 2) for c, v in r.items()} for f, r in R5.items()})

# ------------------------------------------------------------------ R6 GothamDataset2025, both directions
gp = os.path.join(DATA, "gotham_flows.parquet")
if os.path.exists(gp):
    g = pd.read_parquet(gp)
    for c in FEATS:
        if c not in g.columns:
            g[c] = 0.0
    g[FEATS] = g[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    XG, yG, famG = g[FEATS].values.astype(float), g["label"].values, g["family"].values
    gb = np.where(yG == 0)[0].copy(); R.shuffle(gb); calb, testb = gb[: len(gb) // 2], gb[len(gb) // 2:]
    R6 = {"n_gotham_flows": int(len(g)), "n_gotham_benign": int(len(gb)), "gotham_families": dict(collections.Counter(famG))}
    # (a) twin-trained detector (knob regime, all twin data) re-calibrated to 1% FPR on half of the REAL benign
    full = Model("xgb").fit(X, y)
    sg = full.score(XG); thr_real = float(np.quantile(sg[calb], 0.99)); thr_twin = float(np.quantile(OOF["xgb:knob"][0][BEN], 0.99))
    R6["twin_detector_on_gotham"] = {
        "auc": float(roc_auc_score(yG, sg)),
        "at_twin_threshold": {"fpr_real_benign": float(np.mean(sg[testb] > thr_twin)), "recall_real_attack": float(np.mean(sg[yG == 1] > thr_twin))},
        "recalibrated_1pct_on_real_benign": {
            "fpr_real_benign_heldout": float(np.mean(sg[testb] > thr_real)), "recall_real_attack": float(np.mean(sg[yG == 1] > thr_real)),
            "recall_by_family": {k: float(np.mean(sg[(yG == 1) & (famG == k)] > thr_real)) for k in sorted(set(famG[yG == 1]))}}}
    # (b) Gotham-trained detector applied to the twin and its knob library
    sgo = np.zeros(len(XG))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(XG, yG):
        sgo[te] = Model("xgb").fit(XG[tr], yG[tr]).score(XG[te])
    thrG = float(np.quantile(sgo[yG == 0], 0.99))
    st = Model("xgb").fit(XG, yG).score(X)
    R6["gotham_detector_on_twin"] = {
        "gotham_auc_oof": float(roc_auc_score(yG, sgo)), "thr_1pct_real_benign_oof": thrG, "auc_on_twin": float(roc_auc_score(y, st)),
        "fpr_twin_benign": float(np.mean(st[BEN] > thrG)),
        "recall_twin_natural": {f: float(np.mean(st[NATIDX[fam[NATIDX] == f]] > thrG)) for f in FAMS},
        "recall_twin_sobol": {f: float(np.mean(st[SOB[fam[SOB] == f]] > thrG)) for f in FAMS},
        "realised": realised(st, thrG)}
    # (c) benign realism: per-feature KS of twin benign vs real Gotham benign (uncalibrated v1 background benign ->
    #     calibrated benign) and a twin-vs-real benign discriminator (grouped by twin run, 5-fold)
    from scipy.stats import ks_2samp
    v1p = os.path.join(ROOT, "data", "btwin_v1", "flows.parquet")
    gbX = XG[yG == 0]; ks = {"after": {}, "before": {}}
    srcs = {"after": X[BEN]}
    if os.path.exists(v1p):
        v1b = pd.read_parquet(v1p); v1b = v1b[v1b["label"] == 0]
        for c in FEATS:
            if c not in v1b.columns:
                v1b[c] = 0.0
        vb = v1b[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
        srcs["before"] = vb[R.choice(len(vb), min(5000, len(vb)), replace=False)]
    for k_, A in srcs.items():
        for j, c in enumerate(FEATS):
            if np.std(A[:, j]) + np.std(gbX[:, j]) > 0:
                ks[k_][c] = float(ks_2samp(A[:, j], gbX[:, j]).statistic)
    Xd = np.r_[X[BEN], gbX]; yd = np.r_[np.zeros(len(BEN)), np.ones(len(gbX))]
    sd_ = np.zeros(len(yd))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, yd):
        sd_[te] = Model("xgb").fit(Xd[tr], yd[tr]).score(Xd[te])
    R6["benign_realism"] = {"ks_median": {k_: float(np.median(list(v.values()))) for k_, v in ks.items() if v},
                            "ks_after": ks["after"], "ks_before": ks["before"],
                            "discriminator_auc": float(roc_auc_score(yd, sd_))}
    out["R6_gotham"] = R6
    log("R6", R6["twin_detector_on_gotham"]["auc"], {k: v for k, v in R6["twin_detector_on_gotham"]["recalibrated_1pct_on_real_benign"].items() if k != "recall_by_family"},
        "| gotham->twin AUC", round(R6["gotham_detector_on_twin"]["auc_on_twin"], 3), "fpr", round(R6["gotham_detector_on_twin"]["fpr_twin_benign"], 3),
        "nat recall", {f: round(v, 2) for f, v in R6["gotham_detector_on_twin"]["recall_twin_natural"].items()})

# ------------------------------------------------------------------ R7 perturb-ability score vs executed reachability
Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
corr = np.nan_to_num(np.corrcoef(Z.T))
CORR_T = 0.9
fwd_idx = [i for i, c in enumerate(FEATS) if c.startswith("src2dst_")]
PS = {}
for i, c in enumerate(FEATS):
    ps1 = 0.0 if c in ("protocol", "ip_version") else 1.0
    pv = len(np.unique(X[:, i]))
    ps2 = 1.0 if pv > 255 else (0.0 if pv < 2 else 0.5 + 0.5 * (pv - 2) / 253)
    cf = int(np.sum(np.abs(corr[i]) >= CORR_T) - 1)
    ps3 = 0.5 + 0.5 / (2 ** max(cf, 0))
    if c.startswith("dst2src_"):
        nf = int(np.sum(np.abs(corr[i, fwd_idx]) >= CORR_T))
        ps4 = 1.0 if nf == 1 else (0.5 if nf > 1 else 0.0)
    else:
        ps4 = 1.0
    ps5 = 0.5 if any(t in c for t in ("_mean_", "_stddev_", "_min_", "_max_")) else 1.0
    subs = [ps1, ps2, ps3, ps4, ps5]
    tot = float(np.prod(subs) ** (1 / 5)) if min(subs) > 0 else 0.0
    PS[c] = {"PS1": ps1, "PS2": ps2, "PS3": ps3, "PS4": ps4, "PS5": ps5, "PS": tot, "n_corr": cf,
             "class": "low" if tot == 0 else ("high" if tot >= 0.87 else "medium")}
# executed reachability: per family, max |Spearman(implemented knob, feature)| over the functional Sobol flows;
# averaged over families (0 when the feature does not vary)
reach = {}
for i, c in enumerate(FEATS):
    vals = []
    for f in FAMS:
        idx = SOB[(fam[SOB] == f) & (func[SOB] == 1)]
        if len(idx) < 20 or np.std(X[idx, i]) == 0:
            vals.append(0.0); continue
        best_ = 0.0
        for k in IMPL[f]:
            kv = df[f"k_{k}"].values[idx].astype(float)
            if np.std(kv) > 0:
                best_ = max(best_, abs(float(np.nan_to_num(spearmanr(kv, X[idx, i]).correlation))))
        vals.append(best_)
    reach[c] = float(np.mean(vals))
ps_v = np.array([PS[c]["PS"] for c in FEATS]); re_v = np.array([reach[c] for c in FEATS])
out["R7_perturbability"] = {"corr_threshold": CORR_T, "per_feature": {c: {**PS[c], "executed_reach": reach[c]} for c in FEATS},
                            "spearman_ps_vs_reach": float(spearmanr(ps_v, re_v).correlation),
                            "class_counts": dict(collections.Counter(PS[c]["class"] for c in FEATS)),
                            "mean_reach_by_class": {k: float(np.mean([reach[c] for c in FEATS if PS[c]["class"] == k]))
                                                    for k in ("low", "medium", "high") if any(PS[c]["class"] == k for c in FEATS)}}
log("R7", out["R7_perturbability"]["class_counts"], "spearman", round(out["R7_perturbability"]["spearman_ps_vs_reach"], 3),
    out["R7_perturbability"]["mean_reach_by_class"])

fn = "b_revision_quick.json" if QUICK else "b_revision.json"
# save R1-R7 before the sweep stage, so a failure in R8 can never lose the expensive analysis
json.dump(out, open(os.path.join(RES, fn), "w"), indent=1, default=float)
log("saved (pre-R8)", fn)

# ------------------------------------------------------------------ R8 targeted one-factor-at-a-time sweep (held out)
sp = os.path.join(DATA, "sweep_flows.parquet")
try:
    if os.path.exists(sp):
        sw = pd.read_parquet(sp)
        sw = sw[sw["label"] == 1].copy()
        for c in FEATS:
            if c not in sw.columns:
                sw[c] = 0.0
        sw[FEATS] = sw[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        XS = sw[FEATS].values.astype(float)
        runinfo = {}
        rp = os.path.join(DATA, "sweep", "sweep_runs.jsonl")
        for l in open(rp, encoding="utf-8"):
            try:
                r = json.loads(l); runinfo[r["id"]] = r
            except Exception:
                pass
        R8 = {"n_runs": len(runinfo), "n_attack_flows": int(len(sw)), "detectors": {}}
        for key in ("xgb:natural", "xgb:knob", "rf:natural", "mlp:natural", "iso:benign_only"):
            if key not in SC:
                continue
            models, _ = SC[key]; thr = OOF[key][1]
            ens = models[5] if len(models) > 5 else Ensemble(models[:5])
            sw["s"] = ens.score(XS)
            rows = []
            for rid, grp in sw.groupby("run_id"):
                ri = runinfo.get(rid, {})
                rows.append({"run": rid, "family": ri.get("family"), "knob": ri.get("sweep_knob"), "level": ri.get("sweep_level"),
                             "func": int(ri.get("func_ok", 0)), "n_flows": int(len(grp)), "flow_evasion": float(np.mean(grp["s"] <= thr)),
                             "run_evades": int((grp["s"] <= thr).all()), "max_score": float(grp["s"].max())})
            no_flow = sorted(set(runinfo) - set(sw["run_id"]))   # runs whose capture holds no attack flow
            rt = pd.DataFrame(rows)
            rt["func_evade"] = ((rt["run_evades"] == 1) & (rt["func"] == 1)).astype(int)
            agg = (rt.groupby(["family", "knob", "level"], dropna=False)
                     .agg(runs=("run", "size"), func_frac=("func", "mean"), flow_evasion=("flow_evasion", "mean"),
                          runs_evading=("run_evades", "sum"), func_and_evading=("func_evade", "sum"))
                     .reset_index())
            fev = rt[rt["func"] == 1].groupby("family")["flow_evasion"].mean()
            R8["detectors"][key] = {"thr": thr, "per_level": agg.to_dict(orient="records"), "runs_without_attack_flows": no_flow,
                                    "func_and_evading_runs": int(rt["func_evade"].sum()),
                                    "func_runs": int(rt["func"].sum()), "scored_runs": int(len(rt)),
                                    "func_flow_evasion_by_family": {k: float(v) for k, v in fev.items()}}
            log("R8", key, "functional runs", int(rt["func"].sum()), "/", len(rt), "| functional AND fully evading:",
                R8["detectors"][key]["func_and_evading_runs"], "| func flow evasion", {k: round(float(v), 3) for k, v in fev.items()})
        out["R8_sweep"] = R8
        json.dump(out, open(os.path.join(RES, fn), "w"), indent=1, default=float)
        log("saved (with R8)", fn)
except Exception as e:
    import traceback
    log("R8 FAILED (R1-R7 json already saved):", repr(e)[:120]); traceback.print_exc()
