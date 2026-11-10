"""Paper B - failure-case and mechanism analyses behind the realisability gap (feeds the expanded Results).

At the same 1% benign-FPR operating point as b_attack.py, on the same detector and flows:
  A. score margin: how far every FUNCTIONAL executed attack flow stays from the benign region, per family
     (quantiles of the detector score and of the log-odds margin to the threshold);
  B. near misses: the runs whose flows came closest to the threshold, with the knob configuration they used;
     plus the "broken-attack" case: do NON-functional configurations evade? (evasion at the price of function);
  C. knob effects: per family and knob, the functionality rate in the top vs bottom tertile of the knob and the
     rank correlation between the knob value and (i) functionality, (ii) the run's median detector score;
  D. displacement decomposition: which feature groups the unconstrained feature-space attack has to move
     (share of the benign-std-normalised displacement in attacker-forward / whole-flow / victim-side features);
  E. benign realism: per-feature Kolmogorov-Smirnov distance of twin benign vs real Gotham benign, before
     (original twin benign) and after (calibrated benign) calibration;
  F. per-run cost/evasion/functionality points for the trade-off figure.
Writes results/b_failure.json.
"""
import json, os, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from b_attack import load, boundary_attack, CTRL_PREFIX, VICTIM_PREFIX
from b_knobs import NAMES as KNOB_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "btwin"); RES = os.path.join(ROOT, "results")
RNG = np.random.default_rng(0)
FAMS = ["netscan", "svcbrute", "synflood", "coapamp", "httpbeacon"]
FUNC_METRIC = {  # family -> (name, function of run record -> value in [0,1]-ish)
    "netscan": ("ports found / true", lambda r: r.get("ports_found", 0) / max(1, r.get("ports_true", 1))),
    "svcbrute": ("credential found", lambda r: float(r.get("cred_found", 0))),
    "synflood": ("victim goodput loss", lambda r: float(r.get("goodput_drop", 0))),
    "coapamp": ("amplification factor", lambda r: r.get("resp_bytes", 0) / max(1, r.get("req_bytes", 1))),
    "httpbeacon": ("beacons delivered", lambda r: r.get("beacons_ok", 0) / max(1, r.get("beacons_sent", 1))),
}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def group_of(f):
    if f.startswith(VICTIM_PREFIX):
        return "victim (dst2src)"
    if f.startswith("src2dst_"):
        return "attacker forward (src2dst)"
    return "whole-flow (bidirectional)"


def main():
    df, d, feats = load()
    xgbm = d["models"]["xgb"]
    clf = lambda X: xgbm.predict_proba(X)[:, 1]
    benign = df[df["label"] == 0]
    thr = float(np.quantile(clf(benign[feats].values), 0.99))
    decide = lambda X: (clf(X) > thr).astype(int)
    out = {"threshold_fpr1": thr, "families": {}}
    runs = {}
    for l in open(os.path.join(DATA, "b_runs.jsonl"), encoding="utf-8"):
        try:
            r = json.loads(l); runs[r["id"]] = r
        except Exception:
            pass

    s_ben = clf(benign[feats].values)
    out["benign_scores_sample"] = [float(x) for x in RNG.choice(s_ben, min(1500, len(s_ben)), replace=False)]
    bstd = benign[feats].std().replace(0, 1.0).values.astype(float) + 1e-9

    bpool_all = benign[feats].values
    bpool = bpool_all[decide(bpool_all) == 0]
    if len(bpool) > 300:
        bpool = bpool[RNG.choice(len(bpool), 300, replace=False)]

    for fam in FAMS:
        ff = df[(df["family"] == fam) & (df["label"] == 1)]
        if len(ff) == 0:
            continue
        s_all = clf(ff[feats].values)
        func = ff["func_ok"].values == 1
        F = {"n_flows": int(len(ff)), "n_functional_flows": int(func.sum())}
        # ---- A. score margin of functional flows ----
        sf = s_all[func] if func.any() else s_all
        q = lambda a, p: float(np.quantile(a, p)) if len(a) else None
        F["functional_score_quantiles"] = {"min": q(sf, 0), "p1": q(sf, .01), "p5": q(sf, .05), "p50": q(sf, .5)}
        F["functional_logodds_margin"] = {"min": float(np.min(logit(sf) - logit(thr))) if len(sf) else None,
                                          "p50": float(np.median(logit(sf) - logit(thr))) if len(sf) else None}
        F["functional_scores_sample"] = [float(x) for x in RNG.choice(sf, min(600, len(sf)), replace=False)] if len(sf) else []
        # broken-attack case: non-functional flows that the detector calls benign
        nf = s_all[~func]
        F["n_nonfunctional_flows"] = int((~func).sum())
        F["nonfunctional_benign_rate"] = float(np.mean(nf <= thr)) if len(nf) else None
        F["nonfunctional_score_sample"] = [float(x) for x in RNG.choice(nf, min(600, len(nf)), replace=False)] if len(nf) else []
        # ---- B. run-level near misses ----
        per_run = []
        for rid, g in ff.groupby("run_id"):
            s = clf(g[feats].values); r = runs.get(rid, {})
            cfg = r.get("cfg", {}); name, fn = FUNC_METRIC[fam]
            per_run.append({"run_id": rid, "func_ok": int(r.get("func_ok", g["func_ok"].iloc[0])),
                            "n_flows": int(len(g)), "min_score": float(s.min()), "median_score": float(np.median(s)),
                            "benign_frac": float(np.mean(s <= thr)), "wall_s": float(r.get("wall_s", np.nan)),
                            "func_metric": float(fn(r)) if r else None, "cfg": cfg})
        pr = pd.DataFrame(per_run)
        F["func_metric_name"] = FUNC_METRIC[fam][0]
        F["runs"] = per_run
        fr = pr[pr["func_ok"] == 1].sort_values("min_score")
        F["near_misses"] = fr.head(5).to_dict("records")
        bro = pr[(pr["func_ok"] == 0) & (pr["benign_frac"] > 0)].sort_values("benign_frac", ascending=False)
        F["broken_evaders"] = bro.head(5).to_dict("records")
        F["n_runs"] = int(len(pr)); F["n_functional_runs"] = int((pr["func_ok"] == 1).sum())
        F["n_broken_runs_with_benign_flow"] = int(len(bro))
        # ---- C. knob effects (over all runs of the family, default-config repeats included) ----
        K = {}
        if len(pr) > 10:
            for k in KNOB_NAMES:
                v = np.array([c.get(k, np.nan) for c in pr["cfg"]], dtype=float)
                ok = ~np.isnan(v)
                if ok.sum() < 10 or np.std(v[ok]) == 0:
                    continue
                lo, hi = np.quantile(v[ok], [1 / 3, 2 / 3])
                f_lo = float(pr["func_ok"].values[ok][v[ok] <= lo].mean()) if (v[ok] <= lo).any() else None
                f_hi = float(pr["func_ok"].values[ok][v[ok] >= hi].mean()) if (v[ok] >= hi).any() else None
                rf = spearmanr(v[ok], pr["func_ok"].values[ok]).correlation if pr["func_ok"].values[ok].std() > 0 else 0.0
                rs = spearmanr(v[ok], pr["median_score"].values[ok]).correlation
                K[k] = {"func_rate_low_tertile": f_lo, "func_rate_high_tertile": f_hi,
                        "spearman_knob_vs_func": float(np.nan_to_num(rf)), "spearman_knob_vs_score": float(np.nan_to_num(rs))}
        F["knob_effects"] = K
        # ---- D. displacement decomposition of the unconstrained feature-space attack ----
        det = ff[decide(ff[feats].values) == 1]
        samp = det.sample(min(40, len(det)), random_state=0)[feats].values if len(det) else np.zeros((0, len(feats)))
        disp = np.zeros(len(feats)); nsucc = 0
        for x in samp:
            best = None; bestd = 1e18
            for t in RNG.choice(len(bpool), min(6, len(bpool)), replace=False):
                xa = boundary_attack(decide, x, bpool[t], feats)
                if xa is not None:
                    dd = float(np.sum((xa - x) ** 2))
                    if dd < bestd:
                        bestd = dd; best = xa
            if best is not None:
                disp += np.abs(best - x) / bstd; nsucc += 1
        if nsucc:
            disp /= nsucc
            tot = disp.sum() + 1e-12
            share = collections.defaultdict(float)
            for i, f in enumerate(feats):
                share[group_of(f)] += float(disp[i] / tot)
            top = sorted(zip(feats, disp), key=lambda z: -z[1])[:8]
            F["displacement"] = {"n": nsucc, "group_share": dict(share),
                                 "top_features": [(f, float(v)) for f, v in top]}
        # ---- F. trade-off points ----
        F["tradeoff"] = [{"rate_pps": float(r["cfg"].get("rate_pps", np.nan)), "wall_s": r["wall_s"],
                          "min_score": r["min_score"], "func_ok": r["func_ok"], "func_metric": r["func_metric"]}
                         for r in per_run]
        out["families"][fam] = F
        print(fam, {k: F[k] for k in ("n_functional_flows", "functional_score_quantiles", "nonfunctional_benign_rate",
                                       "n_broken_runs_with_benign_flow")}, flush=True)
        if "displacement" in F:
            print("   displacement share", {k: round(v, 3) for k, v in F["displacement"]["group_share"].items()}, flush=True)

    # ---- E. benign realism before/after calibration (per-feature KS vs real Gotham benign) ----
    gp = os.path.join(DATA, "gotham_flows.parquet"); fp = os.path.join(DATA, "flows.parquet")
    if os.path.exists(gp):
        g = pd.read_parquet(gp); gb = g[g["label"] == 0]
        raw = pd.read_parquet(fp); rb = raw[raw["label"] == 0]
        cal = benign
        ks = {"before": {}, "after": {}}
        for f in feats:
            if f not in gb.columns:
                continue
            gv = gb[f].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
            for key, src in (("before", rb), ("after", cal)):
                if f in src.columns:
                    sv = src[f].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(float)
                    if len(sv) > 5000:
                        sv = RNG.choice(sv, 5000, replace=False)
                    if np.std(sv) + np.std(gv) > 0:          # same rule as b_gotham.py (skip constant features)
                        ks[key][f] = float(ks_2samp(sv, gv).statistic)
        out["benign_ks"] = {k: v for k, v in ks.items()}
        out["benign_ks_median"] = {k: float(np.median(list(v.values()))) for k, v in ks.items() if v}
        print("KS median before/after:", out["benign_ks_median"], flush=True)

    json.dump(out, open(os.path.join(RES, "b_failure.json"), "w"), indent=1, default=float)
    print("saved b_failure.json", flush=True)


if __name__ == "__main__":
    main()
