"""Paper B - the feature-space vs traffic-realised evasion gap.

For each attack family, at a fixed detector operating point (benign FPR = 1%):
  FEATURE-SPACE arm: starting from detected attack flows, a decision-based boundary search (unconstrained,
      and a domain-masked variant that may only move attacker-controllable features) finds adversarial
      feature vectors classified benign. ASR_fs = fraction pushed benign.
  FEASIBILITY of those adversarial points (three-way): INFEASIBLE (violates flow invariants, e.g.
      min_ps<=mean_ps<=max_ps, bidir_packets>=src2dst_packets, sums<=totals), NOT-CONTROLLABLE (only moves
      victim-side / dst2src features), or REACHABLE.
  REALISED arm: among the executed knob-library flows of that family that are still FUNCTIONAL (func_ok=1),
      - random-knob baseline: fraction classified benign by the real detector;
      - surrogate transfer (headline): pick the config minimising a surrogate detector (trained on a
        disjoint config split), measure the REAL detector -> realised evasion with no feedback;
      - query-feedback (upper bound): best real-detector evasion over the library.
  GAP = ASR_fs - realised(surrogate), per family. Noise floor tau from repeated default configs.
Writes results/b_attack.json.
"""
import json, os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
try:
    from art.estimators.classification import BlackBoxClassifier
    from art.attacks.evasion import HopSkipJump
    _ART = True
except Exception:
    _ART = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "btwin"); RES = os.path.join(ROOT, "results"); MOD = os.path.join(ROOT, "models")
RNG = np.random.default_rng(0)
# attacker-controllable feature groups: forward direction + overall timing/size the attacker originates.
CTRL_PREFIX = ("src2dst_", "bidirectional_duration", "bidirectional_packets", "bidirectional_bytes",
               "bidirectional_min_piat", "bidirectional_mean_piat", "bidirectional_max_piat",
               "bidirectional_stddev_piat", "src2dst_", "bidirectional_syn", "bidirectional_rst")
VICTIM_PREFIX = ("dst2src_",)


def invariants_ok(x, feats):
    """Cheap flow-feature invariants; violation => the adversarial point is physically impossible."""
    g = {f: x[i] for i, f in enumerate(feats)}
    def le(a, b):
        return not (a in g and b in g) or g[a] <= g[b] + 1e-6
    ok = True
    for pfx in ("bidirectional", "src2dst", "dst2src"):
        ok &= le(f"{pfx}_min_ps", f"{pfx}_mean_ps") and le(f"{pfx}_mean_ps", f"{pfx}_max_ps")
        ok &= le(f"{pfx}_min_piat_ms", f"{pfx}_mean_piat_ms") and le(f"{pfx}_mean_piat_ms", f"{pfx}_max_piat_ms")
    for f in feats:
        if ("_packets" in f or "_bytes" in f or "_ps" in f or "_piat" in f) and g.get(f, 0) < -1e-6:
            ok = False
    if "src2dst_packets" in g and "bidirectional_packets" in g:
        ok &= g["src2dst_packets"] <= g["bidirectional_packets"] + 1e-6
    return ok


def boundary_attack(decide, x, benign_ref, feats, mask=None, steps=30):
    """Decision-based line search toward a benign reference; returns a minimal adversarial x' (decide==0) or None.
    `decide(X)` returns 1 for 'attack', 0 for 'benign' at the detector's operating point."""
    x = x.astype(float).copy()
    target = benign_ref.copy()
    if mask is not None:
        target = x.copy(); target[mask] = benign_ref[mask]      # only move allowed features
    if decide(np.array([target]))[0] != 0:                      # the fully-benign endpoint must evade
        return None
    lo, hi = 0.0, 1.0
    for _ in range(steps):                                       # binary search to the boundary
        mid = (lo + hi) / 2
        if decide(np.array([x + mid * (target - x)]))[0] == 0:
            hi = mid
        else:
            lo = mid
    return x + hi * (target - x)


def load():
    df = pd.read_parquet(os.path.join(DATA, "flows.parquet"))
    cal_path = os.path.join(DATA, "benign_cal_flows.parquet")
    if os.path.exists(cal_path):
        cal = pd.read_parquet(cal_path)
        for c in df.columns:
            if c not in cal.columns:
                cal[c] = np.nan
        df = pd.concat([df[df["label"] == 1], cal.assign(label=0)[df.columns]], ignore_index=True)
    d = pickle.load(open(os.path.join(MOD, "b_detect.pkl"), "rb"))
    feats = d["feats"]
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, d, feats


def main():
    df, d, feats = load()
    xgbm = d["models"]["xgb"]
    clf = lambda X: xgbm.predict_proba(X)[:, 1]
    benign = df[df["label"] == 0]
    # operating threshold at 1% benign FPR
    thr = float(np.quantile(clf(benign[feats].values), 0.99))
    decide = lambda X: (clf(X) > thr).astype(int)       # strict, so ties at the quantile do not flag all benign
    # pool of REAL benign flows that the detector actually classifies benign (valid attack endpoints)
    bpool_all = benign[feats].values
    bpool = bpool_all[decide(bpool_all) == 0]
    if len(bpool) > 300:
        bpool = bpool[RNG.choice(len(bpool), 300, replace=False)]
    fams = sorted(set(df[df["label"] == 1]["family"]))
    mask_ctrl = np.array([f.startswith(CTRL_PREFIX) for f in feats])

    def attack_one(x, mask, tries=6):
        """Line-search toward several real benign flows; return the minimal-perturbation evasive x' or None."""
        best = None; bestd = 1e18
        for t in RNG.choice(len(bpool), min(tries, len(bpool)), replace=False):
            xa = boundary_attack(decide, x, bpool[t], feats, mask=mask)
            if xa is not None and decide(np.array([xa]))[0] == 0:
                dd = float(np.sum(((xa - x)) ** 2))
                if dd < bestd:
                    bestd = dd; best = xa
        return best

    # noise floor: dispersion of default-config flows per family (repeated default runs)
    out = {"threshold_fpr1": thr, "families": {}, "n_flows": len(df)}
    for fam in fams:
        fam_flows = df[(df["family"] == fam) & (df["label"] == 1)]
        detected = fam_flows[decide(fam_flows[feats].values) == 1]
        if len(detected) < 5:
            continue
        samp = detected.sample(min(120, len(detected)), random_state=0)
        Xs = samp[feats].values
        # ---- feature-space arm (unconstrained + domain-masked) ----
        fs_unc = fs_msk = 0; feas = {"infeasible": 0, "not_controllable": 0, "reachable": 0}
        for x in Xs:
            xa = attack_one(x, mask=None)
            if xa is not None:
                fs_unc += 1
                delta = np.abs(xa - x)
                moved = delta > (1e-3 * (np.abs(x) + 1))
                if not invariants_ok(xa, feats):
                    feas["infeasible"] += 1
                elif moved[~mask_ctrl].sum() > 0 and moved[mask_ctrl].sum() == 0:
                    feas["not_controllable"] += 1
                else:
                    feas["reachable"] += 1
            xm = attack_one(x, mask=mask_ctrl)
            if xm is not None:
                fs_msk += 1
        n = len(Xs)
        # strong decision-based attack (HopSkipJump, benign-initialised) -> minimal-perturbation evasion,
        # so the feature-space number is not an artefact of the naive line search.
        hsja_asr = None; hsja_l2 = None; line_l2 = None
        if _ART and len(bpool) and n:
            try:
                Xh = Xs[:30].astype(np.float64)
                init = bpool[RNG.choice(len(bpool), len(Xh))].astype(np.float64)
                bb = BlackBoxClassifier(lambda X: np.eye(2)[decide(np.asarray(X, dtype=np.float64))],
                                        input_shape=(len(feats),), nb_classes=2)
                Xa = HopSkipJump(classifier=bb, targeted=False, norm=2, max_iter=25, max_eval=150,
                                 init_eval=15, verbose=False).generate(x=Xh, x_adv_init=init)
                hsja_asr = float(np.mean(decide(Xa) == 0))
                hsja_l2 = float(np.median(np.linalg.norm(Xa - Xh, axis=1)))
                line_l2 = float(np.median(np.linalg.norm(init - Xh, axis=1)))
            except Exception as e:
                print("hsja err", fam, repr(e)[:80], flush=True)
        # ---- realised arm over the knob library (functional flows only) ----
        func = fam_flows[fam_flows["func_ok"] == 1]
        realised_random = float(np.mean(decide(func[feats].values) == 0)) if len(func) else 0.0
        # surrogate: RF trained on a disjoint half of configs (by run_id), transfer to the real xgb
        runs = sorted(set(fam_flows["run_id"]))
        RNG.shuffle(runs); half = set(runs[:len(runs) // 2])
        sdf = df[df["run_id"].isin(half) | (df["label"] == 0)]
        surro = RandomForestClassifier(n_estimators=200, n_jobs=8, random_state=1,
                                       class_weight="balanced").fit(sdf[feats].values, sdf["label"].values)
        other = func[~func["run_id"].isin(half)]
        realised_transfer = 0.0
        if len(other):
            ss = surro.predict_proba(other[feats].values)[:, 1]
            # pick the config (run) the surrogate thinks is most benign, evaluate on the real detector
            best = other.iloc[ss.argsort()[:max(1, len(other) // 10)]]       # surrogate-preferred configs
            realised_transfer = float(np.mean(decide(best[feats].values) == 0))
        # query-feedback UPPER bound: an attacker that queries the REAL detector and keeps the most-benign
        # functional configurations (strongest possible attacker over the executed library).
        realised_query = 0.0
        if len(func):
            qs = clf(func[feats].values)
            kq = max(1, len(func) // 10)
            bestq = func.iloc[qs.argsort()[:kq]]
            realised_query = float(np.mean(decide(bestq[feats].values) == 0))
        out["families"][fam] = {
            "n_detected": int(len(detected)), "n_attacked": int(n),
            "asr_featurespace_unconstrained": fs_unc / n,
            "asr_featurespace_masked": fs_msk / n,
            "asr_featurespace_hsja": hsja_asr,
            "hsja_median_l2": hsja_l2, "linesearch_median_l2": line_l2,
            "feasibility_of_unconstrained": {k: v / max(1, fs_unc) for k, v in feas.items()},
            "realised_random_knob": realised_random,
            "realised_surrogate_transfer": realised_transfer,
            "realised_query_feedback": realised_query,
            "n_functional_library_flows": int(len(func)),
            "gap_fs_minus_realised": fs_unc / n - realised_transfer,
        }
        print(fam, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["families"][fam].items() if k != "feasibility_of_unconstrained"}, flush=True)

    # ---- measured perturb-ability vs an elShehaby-style a-priori class ----
    # measured: coefficient of variation of each feature across FUNCTIONAL attack flows (what knobs move);
    # a-priori: attacker-controllable (forward/overall) features are "high", victim-side (backward) "low".
    funcall = df[(df["label"] == 1) & (df["func_ok"] == 1)]
    pa = {}
    if len(funcall) > 30:
        for i, f in enumerate(feats):
            v = funcall[f].values.astype(float)
            cv = float(np.std(v) / (abs(np.mean(v)) + 1e-9))
            pa[f] = {"measured_cv": cv, "attacker_controllable": bool(mask_ctrl[i]),
                     "apriori_high": bool(f.startswith(CTRL_PREFIX))}
        ctrl_cv = np.mean([p["measured_cv"] for p in pa.values() if p["attacker_controllable"]])
        vic_cv = np.mean([p["measured_cv"] for p in pa.values() if not p["attacker_controllable"]])
        from scipy.stats import spearmanr
        xs = [1.0 if p["apriori_high"] else 0.0 for p in pa.values()]
        ys = [p["measured_cv"] for p in pa.values()]
        rho = float(spearmanr(xs, ys).correlation) if len(set(xs)) > 1 else 0.0
        out["perturbability"] = {"mean_cv_attacker_controllable": float(ctrl_cv),
                                 "mean_cv_victim_side": float(vic_cv),
                                 "spearman_apriori_vs_measured": rho}
        print("perturbability", {k: round(v, 3) for k, v in out["perturbability"].items()}, flush=True)

    # ---- masking defence: drop the most reachable attacker-controllable features, re-measure ----
    if pa:
        reach = sorted([(f, pa[f]["measured_cv"]) for f in feats if pa[f]["attacker_controllable"]],
                       key=lambda x: -x[1])[:10]
        drop = set(f for f, _ in reach)
        keep_idx = [i for i, f in enumerate(feats) if f not in drop]
        tr = df.sample(frac=0.65, random_state=3); te = df.drop(tr.index)
        md = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                               colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
                               n_jobs=8, random_state=0).fit(tr[feats].values[:, keep_idx], tr["label"].values)
        from sklearn.metrics import roc_auc_score
        au = float(roc_auc_score(te["label"].values, md.predict_proba(te[feats].values[:, keep_idx])[:, 1])) if len(set(te["label"])) > 1 else None
        out["masking_defence"] = {"dropped_features": sorted(drop), "auc_after_masking": au}
        print("masking defence auc", au, flush=True)

    json.dump(out, open(os.path.join(RES, "b_attack.json"), "w"), indent=1)
    print("saved b_attack.json", flush=True)


if __name__ == "__main__":
    main()
