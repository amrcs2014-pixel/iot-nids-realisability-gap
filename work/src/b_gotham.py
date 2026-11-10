"""Paper B external validity: cross-train between the Containernet twin and real GothamDataset2025 flows,
and measure how similar the two benign distributions are. Addresses the same-emulator circularity concern.

  G1  does a detector trained on the TWIN catch REAL Gotham attacks? (recall on Gotham attack flows) and
      what is its false-positive rate on real Gotham benign? -> external validity of the twin detector.
  G2  does a detector trained on real GOTHAM catch the TWIN attacks? (recall on twin attack flows).
  G3  twin-benign vs Gotham-benign similarity: median per-feature KS distance, and a twin-vs-Gotham
      benign discriminator AUROC (0.5 = indistinguishable, 1.0 = trivially separable).
Writes results/b_gotham.json.
"""
import json, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "btwin"); RES = os.path.join(ROOT, "results"); MOD = os.path.join(ROOT, "models")


def XGB():
    return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.8, eval_metric="logloss", tree_method="hist", n_jobs=8, random_state=0)


def main():
    d = pickle.load(open(os.path.join(MOD, "b_detect.pkl"), "rb")); feats = d["feats"]
    twin = pd.read_parquet(os.path.join(DATA, "flows.parquet"))
    goth = pd.read_parquet(os.path.join(DATA, "gotham_flows.parquet"))
    # use the Gotham-calibrated benign for the twin benign class when available
    cal_path = os.path.join(DATA, "benign_cal_flows.parquet")
    calibrated = os.path.exists(cal_path)
    cal = pd.read_parquet(cal_path) if calibrated else None
    dfs = [twin, goth] + ([cal] if calibrated else [])
    for df in dfs:
        for f in feats:
            if f not in df.columns:
                df[f] = 0.0
        df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = {"n_gotham": len(goth), "gotham_families": goth["family"].value_counts().to_dict(),
           "benign_calibrated": bool(calibrated)}

    tb = cal if calibrated else twin[twin["label"] == 0]      # twin benign (calibrated if available)
    ta = twin[twin["label"] == 1]
    gb = goth[goth["label"] == 0]; ga = goth[goth["label"] == 1]
    # twin training set = (calibrated) benign + twin attacks
    twin = pd.concat([tb.assign(label=0), ta.assign(label=1)], ignore_index=True)

    def thr_at(scores_ben, fpr=0.01):
        return float(np.quantile(scores_ben, 1 - fpr))

    # ---- G1: twin-trained detector on real Gotham ----
    m = XGB().fit(twin[feats].values, twin["label"].values)
    s_gb = m.predict_proba(gb[feats].values)[:, 1]
    s_ga = m.predict_proba(ga[feats].values)[:, 1]
    thr = thr_at(m.predict_proba(tb[feats].values)[:, 1], 0.01)   # threshold from TWIN benign (deployment-style)
    out["G1_twin_detector_on_gotham"] = {
        "recall_gotham_attack_at_twinthr": float(np.mean(s_ga > thr)),
        "fpr_gotham_benign_at_twinthr": float(np.mean(s_gb > thr)),
        "recall_by_family": {fam: float(np.mean(m.predict_proba(ga[ga["family"] == fam][feats].values)[:, 1] > thr))
                             for fam in sorted(set(ga["family"]))},
        "auc": float(roc_auc_score(np.r_[np.zeros(len(gb)), np.ones(len(ga))],
                                   np.r_[s_gb, s_ga])) if len(ga) and len(gb) else None}
    print("G1 twin->gotham", {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in out["G1_twin_detector_on_gotham"].items() if k != "recall_by_family"}, flush=True)
    print("   by family", {k: round(v, 3) for k, v in out["G1_twin_detector_on_gotham"]["recall_by_family"].items()}, flush=True)

    # ---- G2: Gotham-trained detector on the twin ----
    if len(ga) > 20:
        mg = XGB().fit(goth[feats].values, goth["label"].values)
        s_ta = mg.predict_proba(ta[feats].values)[:, 1]
        thr_g = thr_at(mg.predict_proba(gb[feats].values)[:, 1], 0.01)
        out["G2_gotham_detector_on_twin"] = {
            "recall_twin_attack_at_gothamthr": float(np.mean(s_ta > thr_g)),
            "fpr_twin_benign": float(np.mean(mg.predict_proba(tb[feats].values)[:, 1] > thr_g)),
            "recall_by_family": {fam: float(np.mean(mg.predict_proba(ta[ta["family"] == fam][feats].values)[:, 1] > thr_g))
                                 for fam in sorted(set(ta["family"]))}}
        print("G2 gotham->twin", {k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in out["G2_gotham_detector_on_twin"].items() if k != "recall_by_family"}, flush=True)

    # ---- G3: benign similarity ----
    ks = []
    for f in feats:
        a = tb[f].values; b = gb[f].values
        if np.std(a) + np.std(b) > 0:
            ks.append(ks_2samp(a, b).statistic)
    Xd = np.r_[tb[feats].values, gb[feats].values]
    yd = np.r_[np.zeros(len(tb)), np.ones(len(gb))]
    perm = np.random.default_rng(0).permutation(len(yd))
    cut = int(0.7 * len(yd)); tr, te = perm[:cut], perm[cut:]
    disc = XGB().fit(Xd[tr], yd[tr])
    out["G3_benign_similarity"] = {
        "median_ks": float(np.median(ks)), "mean_ks": float(np.mean(ks)),
        "discriminator_auc": float(roc_auc_score(yd[te], disc.predict_proba(Xd[te])[:, 1]))}
    print("G3 benign similarity", {k: round(v, 3) for k, v in out["G3_benign_similarity"].items()}, flush=True)

    json.dump(out, open(os.path.join(RES, "b_gotham.json"), "w"), indent=1)
    print("saved b_gotham.json", flush=True)


if __name__ == "__main__":
    main()
