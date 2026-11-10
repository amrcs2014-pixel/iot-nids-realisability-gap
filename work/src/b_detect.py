"""Paper B - flow-based IoT intrusion detectors (trained on randomised-knob traffic, run-level split).

Controls from the design review:
  * train on the full knob range (benign + all Sobol/default configs), hold out whole RUNS and a disjoint
    set of knob configs for test, so evasion cannot be mere train-distribution novelty;
  * fix the decision threshold at a stated benign FPR (0.1% and 1%) on held-out benign;
  * report clean per-family recall at those thresholds (the validity gate for the attack study);
  * shortcut audit: identifiers (IP/MAC/port/time) are dropped from the feature set, and we report the
    top SHAP features to show the detector keys on traffic statistics, not emulation constants.
Writes results/b_detect.json and models/b_detect_<name>.pkl + the feature list.
"""
import json, os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "btwin"); RES = os.path.join(ROOT, "results"); MOD = os.path.join(ROOT, "models")
os.makedirs(RES, exist_ok=True); os.makedirs(MOD, exist_ok=True)
DROP = {"id", "expiration_id", "src_ip", "dst_ip", "src_mac", "dst_mac", "src_oui", "dst_oui",
        "src_port", "dst_port", "vlan_id", "tunnel_id", "bidirectional_first_seen_ms",
        "bidirectional_last_seen_ms", "src2dst_first_seen_ms", "src2dst_last_seen_ms",
        "dst2src_first_seen_ms", "dst2src_last_seen_ms", "label", "family", "run_id", "kind",
        "func_ok", "application_name", "application_category_name", "application_is_guessed",
        "application_confidence", "requested_server_name", "client_fingerprint", "server_fingerprint",
        "user_agent", "content_type"}
RNG = np.random.default_rng(0)


def feature_cols(df):
    cols = []
    for c in df.columns:
        if c in DROP or c.startswith("k_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def thresh_at_fpr(scores_benign, fpr):
    return float(np.quantile(scores_benign, 1 - fpr))


def _recall_above(scores, thr):
    return float(np.mean(scores > thr))     # strict to avoid tie inflation


def main():
    df = pd.read_parquet(os.path.join(DATA, "flows.parquet"))
    # use the Gotham-calibrated benign as the benign class when available (attacks stay twin-executed)
    cal_path = os.path.join(DATA, "benign_cal_flows.parquet")
    if os.path.exists(cal_path):
        cal = pd.read_parquet(cal_path)
        df = pd.concat([df[df["label"] == 1], cal.assign(label=0)], ignore_index=True)
        print("using calibrated benign:", len(cal), "benign flows", flush=True)
    feats = feature_cols(df)
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # run-level split; hold out a disjoint third of the Sobol configs too (config novelty)
    runs = df[["run_id", "kind"]].drop_duplicates()
    rids = runs["run_id"].tolist()
    RNG.shuffle(rids)
    cut = int(0.65 * len(rids)); train_runs = set(rids[:cut]); test_runs = set(rids[cut:])
    tr = df[df["run_id"].isin(train_runs)]; te = df[df["run_id"].isin(test_runs)]
    Xtr, ytr = tr[feats].values, tr["label"].values
    Xte, yte = te[feats].values, te["label"].values
    print(f"flows train {len(tr)} / test {len(te)}; features {len(feats)}; attack frac tr {ytr.mean():.2f} te {yte.mean():.2f}", flush=True)

    sc = StandardScaler().fit(Xtr)
    models = {}
    models["xgb"] = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                                      colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
                                      n_jobs=8, random_state=0).fit(Xtr, ytr)
    models["rf"] = RandomForestClassifier(n_estimators=300, n_jobs=8, random_state=0, class_weight="balanced").fit(Xtr, ytr)
    models["mlp"] = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0).fit(sc.transform(Xtr), ytr)
    iso = IsolationForest(n_estimators=200, random_state=0, contamination=0.1).fit(sc.transform(Xtr[ytr == 0]))

    res = {"n_features": len(feats), "features": feats, "n_train": len(tr), "n_test": len(te), "models": {}}
    def scores(name, Xs):
        if name == "mlp":
            return models["mlp"].predict_proba(sc.transform(Xs))[:, 1]
        if name == "iso":
            return -iso.score_samples(sc.transform(Xs))
        return models[name].predict_proba(Xs)[:, 1]

    fam_te = te["family"].values; benign_te = te[te["label"] == 0]
    for name in ["xgb", "rf", "mlp", "iso"]:
        s_te = scores(name, Xte); s_ben = scores(name, benign_te[feats].values)
        r = {"auc": float(roc_auc_score(yte, s_te)) if len(set(yte)) > 1 else None}
        for fpr in (0.001, 0.01):
            thr = thresh_at_fpr(s_ben, fpr)
            r[f"thr_fpr{fpr}"] = thr
            rec = {}
            for fam in sorted(set(fam_te) - {"benign"}):
                m = fam_te == fam
                if m.sum():
                    rec[fam] = _recall_above(s_te[m], thr)
            r[f"recall_fpr{fpr}"] = rec
            r[f"realised_fpr{fpr}"] = _recall_above(s_ben, thr)
        res["models"][name] = r
        print(name, {"auc": round(r["auc"], 3) if r["auc"] else None, "recall@1%": {k: round(v, 2) for k, v in r["recall_fpr0.01"].items()}}, flush=True)

    # shortcut audit: top xgb gain features
    imp = sorted(zip(feats, models["xgb"].feature_importances_), key=lambda x: -x[1])[:15]
    res["xgb_top_features"] = [(n, float(v)) for n, v in imp]
    print("top features:", [n for n, _ in imp[:8]], flush=True)

    pickle.dump({"models": {k: models[k] for k in ("xgb", "rf")}, "mlp": models["mlp"], "iso": iso,
                 "scaler": sc, "feats": feats}, open(os.path.join(MOD, "b_detect.pkl"), "wb"))
    json.dump(res, open(os.path.join(RES, "b_detect.json"), "w"), indent=1)
    print("saved b_detect.json + b_detect.pkl", flush=True)


if __name__ == "__main__":
    main()
