"""Paper B - figures, number macros and tables from results/b_*.json (+ b_runs.jsonl) into paperB/.

Figures (all vector PDF, regenerated from the result files):
  b_fig_pipeline  methodology diagram (twin -> knob library -> flows -> detectors -> two arms -> gap, + validity loop)
  b_fig_gap       (a) feature-space vs realised evasion per family, all attacker variants; (b) HSJA vs line-search L2
  b_fig_margin    failure cases: detector log-odds of benign vs functional (and broken) attack flows, per family
  b_fig_knobs     knob -> functionality and knob -> detector-score rank correlations (13 knobs x 5 families)
  b_fig_tradeoff  per-run evasion vs functionality: the empty "functional and benign" quadrant
  b_fig_mechanism (a) which feature groups the feature-space attack must move; (b) detector feature importance
  b_fig_gotham    (a) per-feature KS to real Gotham benign before/after calibration; (b) transfer to real Gotham
Tables: tab_detect (all detectors), tab_knobs (catalogue), tab_attack (per-family summary), tab_nearmiss (closest runs).
"""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); DATA = os.path.join(ROOT, "data", "btwin"); OUT = os.path.join(ROOT, "paperB")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9, "legend.fontsize": 7.5})
FAMS = ["netscan", "svcbrute", "synflood", "coapamp", "httpbeacon"]
NICE = {"netscan": "scan", "svcbrute": "brute", "synflood": "flood", "coapamp": "coap-amp", "httpbeacon": "beacon"}
C = {"fs": "#E45756", "hsja": "#F58518", "masked": "#B279A2", "rand": "#54A24B", "transfer": "#4C78A8",
     "query": "#72B7B2", "benign": "#9D9D9D", "attacker": "#4C78A8", "whole": "#F58518", "victim": "#E45756"}
KNOB_DESC = [  # (name, range, what the attacker sets)
    ("rate\\_pps", "1--400 (log)", "offered send rate (packets/s)"),
    ("ipd\\_jitter", "0--1", "uniform jitter as a fraction of the mean inter-packet delay"),
    ("pad\\_bytes", "0--512", "payload padding appended to every packet"),
    ("pkts\\_per\\_conn", "1--40", "packets sent before opening a new connection (TCP families)"),
    ("parallel", "1--8", "concurrent streams / sources"),
    ("srcport\\_mode", "\\{fixed, sequential, random-high, random-any\\}", "source-port strategy"),
    ("ttl", "32--128", "IP time-to-live stamped on outgoing packets"),
    ("tcp\\_window", "512--65535 (log)", "advertised TCP receive window"),
    ("frag\\_mtu", "\\{none, 1500, 576, 256\\}", "IP fragmentation via a reduced path MTU"),
    ("netem\\_delay", "0--100 ms", "egress delay on the attacker NIC (\\texttt{tc netem})"),
    ("netem\\_loss", "0--5\\%", "egress loss on the attacker NIC (\\texttt{tc netem})"),
    ("idle\\_insert", "0--3 s", "idle gap inserted after every burst"),
    ("cover", "0--1", "probability of interleaving benign-looking cover packets"),
]


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def pct(x, nd=1):
    return f"{100*x:.{nd}f}"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def group_of(f):
    if f.startswith("dst2src_"):
        return "victim"
    if f.startswith("src2dst_"):
        return "attacker"
    return "whole"


def box(ax, x, y, w, h, title, body, fc="#F4F6F8", ec="#444444", lw=0.9, tfs=6.6, bfs=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=tfs, fontweight="bold")
    ax.text(x + w / 2, y + 1.6, body, ha="center", va="bottom", fontsize=bfs, linespacing=1.3)


def arrow(ax, p, q, color="#444444", lw=1.0, style="-|>", cs="arc3,rad=0.0", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, connectionstyle=cs, linestyle=ls, shrinkA=0, shrinkB=0))


def fig_pipeline(nruns, nfams, nknobs, nfeat):
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
    Y1, H1 = 51, 14; Y2, H2 = 30, 13; Y3, H3 = 3, 17
    # --- row 1: generation ---
    box(ax, 1, Y1, 21, H1, "Reference traffic", "real GothamDataset2025\nbenign + attack pcaps", fc="#FFF4E5", ec="#C98A1E")
    box(ax, 26, Y1, 22, H1, "IoT digital twin", "Containernet + Open vSwitch\nMQTT, CoAP, HTTP, TCP services\n4 calibrated devices + attacker", fc="#E8F1FA")
    box(ax, 52, Y1, 22, H1, "Attacker knob catalogue", f"{nknobs} host-level knobs\nSobol sample, {120} cfg / family\n+ repeated natural cfg", fc="#F3E8F6")
    box(ax, 78, Y1, 21, H1, "Executed knob library", f"{nfams} families, {nruns} runs\npcap + functionality oracle\n+ attacker cost", fc="#E9F5E9", ec="#3E8E41")
    arrow(ax, (22, Y1 + 7), (26, Y1 + 7), color="#C98A1E"); ax.text(24, Y1 + 8.3, "calibrate", ha="center", fontsize=5.6, color="#C98A1E")
    arrow(ax, (48, Y1 + 7), (52, Y1 + 7)); arrow(ax, (74, Y1 + 7), (78, Y1 + 7))
    # --- row 2: audit, detectors, flows ---
    box(ax, 1, Y2, 21, H2, "External validity", "twin detector $\\leftrightarrow$ real Gotham\nKS distance, discriminator", fc="#FFF4E5", ec="#C98A1E")
    box(ax, 26, Y2, 34, H2, "Flow detectors", "XGBoost, RF, MLP, isolation forest\nrun-level split, held-out knob configs\nthreshold fixed at 1% benign FPR", fc="#F4F6F8")
    box(ax, 64, Y2, 35, H2, "Flow extraction", f"nfstream, {nfeat} statistical features\nidentifiers dropped; labelled by run,\nfamily and functionality", fc="#F4F6F8")
    arrow(ax, (88.5, Y1), (88.5, Y2 + H2)); arrow(ax, (64, Y2 + 6.5), (60, Y2 + 6.5))
    arrow(ax, (11.5, Y1), (11.5, Y2 + H2), color="#C98A1E"); arrow(ax, (26, Y2 + 6.5), (22, Y2 + 6.5), color="#C98A1E", ls="--")
    # --- row 3: two arms converge on the gap ---
    box(ax, 1, Y3, 31, H3, "Feature-space arm", "line search, HopSkipJump,\nattacker-controllable-only (masked)\n$\\rightarrow$ ASR$_\\mathrm{fs}$, feasibility labels,\ndisplacement decomposition", fc="#FBE9E9", ec=C["fs"])
    box(ax, 36, Y3 + 2, 27, H3 - 4, "Realisability gap", "$\\Delta = \\mathrm{ASR}_\\mathrm{fs} - \\mathrm{realised}$\nper family, same detector,\nsame operating point", fc="#FFFFFF", ec="#222222", lw=1.4, tfs=7.6)
    box(ax, 67, Y3, 32, H3, "Realised arm", "functional library flows only:\nrandom knob, surrogate transfer\n(no feedback), query-feedback bound\n$\\rightarrow$ realised evasion", fc="#E6EEF7", ec=C["transfer"])
    arrow(ax, (36, Y2), (18, Y3 + H3), color=C["fs"]); arrow(ax, (52, Y2), (80, Y3 + H3), color=C["transfer"])
    arrow(ax, (84, Y2), (84, Y3 + H3), color="#3E8E41"); ax.text(85.2, Y2 - 4.2, "functional flows", fontsize=5.6, color="#3E8E41", ha="left")
    arrow(ax, (32, Y3 + H3 / 2), (36, Y3 + H3 / 2), color=C["fs"], lw=1.3); arrow(ax, (67, Y3 + H3 / 2), (63, Y3 + H3 / 2), color=C["transfer"], lw=1.3)
    fig.savefig(os.path.join(OUT, "b_fig_pipeline.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_gap(fams):
    present = [f for f in FAMS if f in fams]
    x = np.arange(len(present))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [2.3, 1]})
    series = [("asr_featurespace_unconstrained", "feature-space (line search)", C["fs"]),
              ("asr_featurespace_hsja", "feature-space (HopSkipJump)", C["hsja"]),
              ("asr_featurespace_masked", "feature-space, attacker-controllable only", C["masked"]),
              ("realised_random_knob", "realised: random knob", C["rand"]),
              ("realised_surrogate_transfer", "realised: surrogate transfer", C["transfer"]),
              ("realised_query_feedback", "realised: query-feedback bound", C["query"])]
    w = 0.13
    for i, (k, lab, col) in enumerate(series):
        v = [100 * (fams[f].get(k) or 0.0) for f in present]
        bars = a.bar(x + (i - 2.5) * w, v, w, label=lab, color=col)
        for bx, vv in zip(bars, v):
            if vv == 0:
                a.plot([bx.get_x() + w / 2], [1.5], marker="v", ms=3, color=col, clip_on=False)
    a.set_xticks(x); a.set_xticklabels([NICE[f] for f in present], rotation=20); a.set_ylim(0, 108)
    a.set_ylabel("evasion at 1% benign FPR (%)")
    a.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=6.6, handlelength=1.2, columnspacing=1.0)
    a.set_title("(a) feature-space vs traffic-realised evasion", loc="left")
    l2h = [fams[f].get("hsja_median_l2") or 0 for f in present]; l2l = [fams[f].get("linesearch_median_l2") or 0 for f in present]
    b.bar(x - 0.18, l2l, 0.36, color=C["fs"], label="line search"); b.bar(x + 0.18, l2h, 0.36, color=C["hsja"], label="HopSkipJump")
    b.set_xticks(x); b.set_xticklabels([NICE[f] for f in present], rotation=30, ha="right", fontsize=7.5)
    b.set_ylabel("median $L_2$ perturbation"); b.set_ylim(0, 1.3 * max(l2l + l2h + [1]))
    b.legend(frameon=False, fontsize=6.6, loc="upper left"); b.set_title("(b) perturbation size", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_gap.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_margin(fail):
    fams = fail["families"]; present = [f for f in FAMS if f in fams]
    thr = logit(fail["threshold_fpr1"]); ben = logit(fail["benign_scores_sample"])
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    rng = np.random.default_rng(1)
    ax.scatter(rng.uniform(-0.3, 0.3, len(ben)), ben, s=4, color=C["benign"], alpha=0.5, lw=0, label="calibrated benign (held-out)")
    for i, f in enumerate(present, start=1):
        sf = logit(fams[f]["functional_scores_sample"]); nf = logit(fams[f].get("nonfunctional_score_sample", []))
        off = 0.17 if len(nf) else 0.0
        if len(nf):
            ax.scatter(i - off + rng.uniform(-0.14, 0.14, len(nf)), nf, s=4, color="#BBBBBB", alpha=0.7, lw=0, label="broken attack (non-functional)" if i == 1 else None)
        ax.scatter(i + off + rng.uniform(-0.14, 0.14, len(sf)), sf, s=4, color=C["transfer"], alpha=0.6, lw=0, label="functional attack flows" if i == 1 else None)
        q = fams[f]["functional_score_quantiles"]
        mn = logit(q["min"])
        ax.plot([i - 0.35, i + 0.35], [mn, mn], color="#222222", lw=1.0)
        ax.annotate(f"closest\n{mn-thr:+.1f}", (i + 0.38, mn), fontsize=6.3, va="center")
    ax.axhline(thr, color=C["fs"], ls="--", lw=1.0); ax.text(0.55, thr + 0.6, "operating threshold (1% benign FPR)", color=C["fs"], fontsize=7)
    ax.set_xticks(range(len(present) + 1)); ax.set_xticklabels(["benign"] + [NICE[f] for f in present])
    ax.set_ylabel("detector log-odds"); ax.legend(frameon=False, loc="lower right", ncol=1, markerscale=2.5)
    ax.set_title("Failure cases: no functional (or broken) executed attack flow reaches the benign region", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_margin.pdf")); plt.close(fig)


def fig_knobs(fail):
    fams = fail["families"]; present = [f for f in FAMS if f in fams]
    knobs = [k for k, _, _ in KNOB_DESC]; raw = [k.replace("\\_", "_") for k in knobs]
    Mf = np.full((len(raw), len(present)), np.nan); Ms = Mf.copy()
    for j, f in enumerate(present):
        K = fams[f].get("knob_effects", {})
        for i, k in enumerate(raw):
            if k in K:
                Mf[i, j] = K[k]["spearman_knob_vs_func"]; Ms[i, j] = K[k]["spearman_knob_vs_score"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.6), sharey=True)
    for ax, M, title in ((axs[0], Mf, "(a) knob $\\rightarrow$ attack still functional"), (axs[1], Ms, "(b) knob $\\rightarrow$ detector score ($+$ = more attack-like)")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(present))); ax.set_xticklabels([NICE[f] for f in present], rotation=30, ha="right")
        ax.set_yticks(range(len(raw))); ax.set_yticklabels(raw, fontsize=7.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]) and abs(M[i, j]) >= 0.3:
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6, color="white" if abs(M[i, j]) > 0.6 else "black")
        for j in range(M.shape[1]):
            if np.all(np.isnan(M[:, j])):
                ax.text(j, M.shape[0] / 2 - 0.5, "always functional", ha="center", va="center", fontsize=6.2, color="#555555", rotation=90)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Spearman rank correlation", fontsize=7.5)
    fig.savefig(os.path.join(OUT, "b_fig_knobs.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_tradeoff(fail):
    fams = fail["families"]; present = [f for f in FAMS if f in fams]; thr = logit(fail["threshold_fpr1"])
    fig, axs = plt.subplots(1, len(present), figsize=(7.1, 2.3), sharey=True)
    for ax, f in zip(axs, present):
        T = fams[f]["tradeoff"]; name = fams[f]["func_metric_name"]
        xs = np.array([t["func_metric"] if t["func_metric"] is not None else np.nan for t in T], dtype=float)
        ys = logit([t["min_score"] for t in T]); ok = np.array([t["func_ok"] for t in T]) == 1
        if f == "coapamp":
            xs = xs / max(1e-9, np.nanmax(xs)); name = "amplification (norm.)"
        ax.scatter(xs[~ok], ys[~ok], s=9, color="#BBBBBB", label="broken", lw=0)
        ax.scatter(xs[ok], ys[ok], s=9, color=C["transfer"], label="functional", lw=0)
        ax.axhline(thr, color=C["fs"], ls="--", lw=0.9)
        lo, hi = -7, 16
        ax.axvspan(0.5 if f != "synflood" else 0.3, 1.05, ymin=0, ymax=(thr - lo) / (hi - lo), color=C["rand"], alpha=0.18, lw=0)
        ax.set_title(NICE[f], fontsize=8.5); ax.set_xlabel(name, fontsize=7); ax.set_ylim(lo, hi)
        ax.set_xlim(-0.05, 1.05) if f != "synflood" else ax.set_xlim(-0.6, 1.05)
    axs[0].set_ylabel("lowest detector log-odds in run"); axs[0].legend(frameon=False, loc="upper left", fontsize=6.5, markerscale=1.6)
    axs[-1].text(0.0, thr - 1.0, "evasion region", color=C["fs"], fontsize=6.3, va="top")
    fig.suptitle("Per-run evasion vs functionality: the 'functional and benign' corner (shaded) is empty for every family", fontsize=8.5, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_tradeoff.pdf")); plt.close(fig)


def fig_mechanism(fail, det):
    fams = fail["families"]; present = [f for f in FAMS if f in fams and "displacement" in fams[f]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
    keys = [("attacker forward (src2dst)", C["attacker"], "attacker-forward (src2dst)"),
            ("whole-flow (bidirectional)", C["whole"], "whole-flow (bidirectional)"),
            ("victim (dst2src)", C["victim"], "victim-side (dst2src)")]
    bottom = np.zeros(len(present))
    for k, col, lab in keys:
        v = np.array([100 * fams[f]["displacement"]["group_share"].get(k, 0) for f in present])
        a.bar(range(len(present)), v, 0.6, bottom=bottom, color=col, label=lab); bottom += v
    a.set_xticks(range(len(present))); a.set_xticklabels([NICE[f] for f in present], rotation=30, ha="right"); a.set_ylabel("share of required displacement (%)")
    a.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    a.set_title("(a) what the feature-space attack must move", loc="left")
    top = det.get("xgb_top_features", [])[:10]
    names = [n for n, _ in top][::-1]; vals = [v for _, v in top][::-1]
    cols = [{"attacker": C["attacker"], "whole": C["whole"], "victim": C["victim"]}[group_of(n)] for n in names]
    b.barh(range(len(names)), vals, color=cols); b.set_yticks(range(len(names))); b.set_yticklabels(names, fontsize=7)
    for i, v in enumerate(vals):
        b.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=6.3)
    b.set_xlim(0, max(vals) * 1.18); b.set_xlabel("XGBoost gain importance (top 10 of %d features)" % det.get("n_features", 0))
    b.set_title("(b) what the detector relies on", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_mechanism.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_gotham(fail, goth):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
    ks = fail.get("benign_ks", {})
    if ks.get("before") and ks.get("after"):
        feats = sorted(ks["after"], key=lambda f: ks["after"][f])
        bef = [ks["before"].get(f, np.nan) for f in feats]; aft = [ks["after"][f] for f in feats]
        a.plot(range(len(feats)), sorted(bef), color=C["fs"], lw=1.4, label=f"uncalibrated twin (median {np.nanmedian(bef):.2f})")
        a.plot(range(len(feats)), aft, color=C["transfer"], lw=1.4, label=f"calibrated twin (median {np.median(aft):.2f})")
        a.fill_between(range(len(feats)), aft, 0, color=C["transfer"], alpha=0.12, lw=0)
        worst = feats[-4:][::-1]
        a.text(0.40, 0.74, "largest residual KS after calibration:\n" + "\n".join(f"{f.replace('bidirectional_', 'bi_').replace('_ms', '')}  {ks['after'][f]:.2f}" for f in worst),
               transform=a.transAxes, fontsize=6.0, va="top", ha="left", color="#333333")
        a.set_xlabel("flow feature (each curve sorted by its KS)"); a.set_ylabel("KS distance to real Gotham benign"); a.set_ylim(0, 1.05)
        a.legend(frameon=False, loc="upper left"); a.set_title("(a) benign realism before / after calibration", loc="left")
    g1 = goth.get("G1_twin_detector_on_gotham", {}); rb = g1.get("recall_by_family", {})
    labs = {"coapamp": "coap-amp", "merlin_c2": "HTTP C2", "mirai_infect": "Mirai infect", "netscan": "scan", "synflood": "flood"}
    order = [k for k in ("coapamp", "merlin_c2", "mirai_infect", "netscan", "synflood") if k in rb]
    vals = [100 * rb[k] for k in order]
    b.bar(range(len(order)), vals, 0.6, color=C["transfer"], label="recall on real Gotham attacks")
    b.bar([len(order)], [100 * g1.get("fpr_gotham_benign_at_twinthr", 0)], 0.6, color=C["benign"], label="FPR on real Gotham benign")
    b.set_xticks(range(len(order) + 1)); b.set_xticklabels([labs[k] for k in order] + ["benign FPR"], fontsize=7, rotation=30, ha="right")
    b.set_ylim(0, 135); b.set_ylabel("% at the twin-calibrated threshold"); b.legend(frameon=False, fontsize=6.4, loc="upper right")
    for i, v in enumerate(vals + [100 * g1.get("fpr_gotham_benign_at_twinthr", 0)]):
        b.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=6.5)
    b.set_title(f"(b) twin detector on real Gotham, AUROC {g1.get('auc', 0):.2f}", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_gotham.pdf"), bbox_inches="tight"); plt.close(fig)


def main():
    det, atk, goth, fail = J("b_detect.json"), J("b_attack.json"), J("b_gotham.json"), J("b_failure.json")
    runs = []
    rp = os.path.join(DATA, "b_runs.jsonl")
    if os.path.exists(rp):
        for l in open(rp, encoding="utf-8"):
            try:
                runs.append(json.loads(l))
            except Exception:
                pass
    M = collections.OrderedDict()
    def macro(name, val, nd=3):
        M[name] = f"{val:.{nd}f}" if isinstance(val, float) else str(val)
    for k in ("nRuns", "nBenignRuns", "nAttackRuns", "nFlows", "nFeatures", "detAuc", "cleanRecall", "isoRecall",
              "asrFsMean", "realisedMean", "randomKnobMean", "gapMean", "gapMax", "feasinfeasible", "feasnotcontrollable",
              "feasreachable", "pertSpearman", "maskAuc", "asrFsMasked", "asrFsHsja", "queryFeedback", "gothRecall",
              "gothFpr", "gothAuc", "gothTwinRecall", "ksMed", "ksBefore", "discAuc", "nGotham", "nKnobs", "nCfg",
              "closestFam", "closestMargin", "closestScore", "medianMargin", "nonfuncBenign", "nFunctionalRuns",
              "dispVictimWhole", "hsjaLtwoRatio", "nGothamBenign", "cvCtrl", "cvVictim", "nCalBenign",
              "rhoRateScan", "rhoRateFlood", "funcScanHighRate", "funcScanLowRate", "rhoPadScan", "rhoPadBrute",
              "rhoPadFlood", "rhoPadBeacon", "rhoSrcportCoap", "topImpOne", "topImpTwo", "topFeatOne", "topFeatTwo"):
        M[k] = "--"
    for fam in FAMS:
        M["margin" + fam] = "--"
    for fam in FAMS:
        M["func" + fam] = "--"
    macro("nKnobs", len(KNOB_DESC)); macro("nCfg", 120)

    if runs:
        atkruns = [r for r in runs if r["kind"] == "attack"]
        macro("nRuns", len(runs)); macro("nBenignRuns", sum(r["kind"] == "benign" for r in runs)); macro("nAttackRuns", len(atkruns))
        fo = collections.defaultdict(list)
        for r in atkruns:
            fo[r["family"]].append(r.get("func_ok", 0))
        for fam in FAMS:
            if fo[fam]:
                macro("func" + fam, pct(np.mean(fo[fam])))
        macro("nFunctionalRuns", int(sum(r.get("func_ok", 0) for r in atkruns)))
    if det:
        macro("nFlows", f"{det['n_train'] + det['n_test']:,}".replace(",", "{,}")); macro("nFeatures", det["n_features"])
        x = det["models"]["xgb"]; macro("detAuc", x["auc"] or 0.0)
        rec = list(x.get("recall_fpr0.01", {}).values())
        if rec:
            macro("cleanRecall", pct(np.mean(rec)))
        iso = det["models"].get("iso", {}).get("recall_fpr0.01", {})
        if iso:
            macro("isoRecall", pct(np.mean(list(iso.values()))))
    if atk and atk.get("families"):
        fams = atk["families"]
        g = lambda k: [v[k] for v in fams.values() if v.get(k) is not None]
        macro("asrFsMean", pct(np.mean(g("asr_featurespace_unconstrained")))); macro("asrFsMasked", pct(np.mean(g("asr_featurespace_masked"))))
        if g("asr_featurespace_hsja"):
            macro("asrFsHsja", pct(np.mean(g("asr_featurespace_hsja"))))
        if g("hsja_median_l2") and g("linesearch_median_l2"):
            macro("hsjaLtwoRatio", float(np.median(np.array(g("hsja_median_l2")) / np.array(g("linesearch_median_l2")))), 2)
        macro("realisedMean", pct(np.mean(g("realised_surrogate_transfer")))); macro("randomKnobMean", pct(np.mean(g("realised_random_knob"))))
        macro("gapMean", pct(np.mean(g("gap_fs_minus_realised")))); macro("gapMax", pct(np.max(g("gap_fs_minus_realised"))))
        if g("realised_query_feedback"):
            macro("queryFeedback", pct(np.mean(g("realised_query_feedback"))))
        if atk.get("perturbability"):
            P = atk["perturbability"]; macro("pertSpearman", P["spearman_apriori_vs_measured"], 2)
            macro("cvCtrl", P["mean_cv_attacker_controllable"], 1); macro("cvVictim", P["mean_cv_victim_side"], 1)
        if atk.get("masking_defence", {}).get("auc_after_masking"):
            macro("maskAuc", atk["masking_defence"]["auc_after_masking"])
        feas = collections.defaultdict(list)
        for v in fams.values():
            for k, val in v.get("feasibility_of_unconstrained", {}).items():
                feas[k].append(val)
        for k in ("infeasible", "not_controllable", "reachable"):
            if feas[k]:
                macro("feas" + k.replace("_", ""), pct(np.mean(feas[k])))
    if goth:
        macro("nGotham", f"{goth.get('n_gotham', 0):,}".replace(",", "{,}"))
        macro("nGothamBenign", f"{goth.get('gotham_families', {}).get('benign', 0):,}".replace(",", "{,}"))
        g1 = goth.get("G1_twin_detector_on_gotham", {})
        if g1:
            macro("gothRecall", pct(g1.get("recall_gotham_attack_at_twinthr", 0))); macro("gothFpr", pct(g1.get("fpr_gotham_benign_at_twinthr", 0)))
            if g1.get("auc") is not None:
                macro("gothAuc", g1["auc"])
        g2 = goth.get("G2_gotham_detector_on_twin", {})
        if g2:
            macro("gothTwinRecall", pct(g2.get("recall_twin_attack_at_gothamthr", 0)))
        g3 = goth.get("G3_benign_similarity", {})
        if g3:
            macro("ksMed", g3.get("median_ks", 0), 2); macro("discAuc", g3.get("discriminator_auc", 0))
    if fail:
        if fail.get("benign_ks_median"):
            macro("ksBefore", fail["benign_ks_median"].get("before", 0), 2)
        thr = logit(fail["threshold_fpr1"]); best = None
        for f, F in fail["families"].items():
            m = F["functional_logodds_margin"]["min"]
            if m is not None and (best is None or m < best[1]):
                best = (f, m, F["functional_score_quantiles"]["min"])
        if best:
            macro("closestFam", NICE[best[0]]); macro("closestMargin", best[1], 1); macro("closestScore", best[2], 3)
        macro("medianMargin", float(np.median([F["functional_logodds_margin"]["p50"] for F in fail["families"].values() if F["functional_logodds_margin"]["p50"] is not None])), 1)
        nb = [F["nonfunctional_benign_rate"] for F in fail["families"].values() if F.get("nonfunctional_benign_rate") is not None]
        if nb:
            macro("nonfuncBenign", pct(np.mean(nb)))
        vw = [100 * (F["displacement"]["group_share"].get("victim (dst2src)", 0) + F["displacement"]["group_share"].get("whole-flow (bidirectional)", 0))
              for F in fail["families"].values() if "displacement" in F]
        if vw:
            macro("dispVictimWhole", f"{min(vw):.0f}--{max(vw):.0f}")
        for f, F in fail["families"].items():
            m = F["functional_logodds_margin"]["min"]
            if m is not None:
                macro("margin" + f, f"{m:+.1f}")
        def rho(fam, knob, key):
            v = fail["families"].get(fam, {}).get("knob_effects", {}).get(knob)
            return f"{v[key]:+.2f}" if v else "--"
        macro("rhoRateScan", rho("netscan", "rate_pps", "spearman_knob_vs_func")); macro("rhoRateFlood", rho("synflood", "rate_pps", "spearman_knob_vs_func"))
        for nm, fam in (("Scan", "netscan"), ("Brute", "svcbrute"), ("Flood", "synflood"), ("Beacon", "httpbeacon")):
            macro("rhoPad" + nm, rho(fam, "pad_bytes", "spearman_knob_vs_score"))
        macro("rhoSrcportCoap", rho("coapamp", "srcport_mode", "spearman_knob_vs_score"))
        ke = fail["families"].get("netscan", {}).get("knob_effects", {}).get("rate_pps")
        if ke:
            macro("funcScanHighRate", pct(ke["func_rate_high_tertile"], 0)); macro("funcScanLowRate", pct(ke["func_rate_low_tertile"], 0))
    if det and det.get("xgb_top_features"):
        (f1, v1), (f2, v2) = det["xgb_top_features"][:2]
        macro("topFeatOne", f1.replace("_", "\\_")); macro("topImpOne", v1, 2); macro("topFeatTwo", f2.replace("_", "\\_")); macro("topImpTwo", v2, 2)
    cal_path = os.path.join(DATA, "benign_cal_flows.parquet")
    if os.path.exists(cal_path):
        import pandas as pd
        macro("nCalBenign", f"{len(pd.read_parquet(cal_path)):,}".replace(",", "{,}"))
    with open(os.path.join(OUT, "numbers_B.tex"), "w", encoding="ascii") as fh:
        for k, v in M.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    # ---------------- figures ----------------
    fig_pipeline(M["nRuns"], 5, len(KNOB_DESC), M["nFeatures"])
    if atk and atk.get("families"):
        fig_gap(atk["families"])
    if fail:
        fig_margin(fail); fig_knobs(fail); fig_tradeoff(fail)
        if det:
            fig_mechanism(fail, det)
        if goth:
            fig_gotham(fail, goth)

    # ---------------- tables ----------------
    if det:
        with open(os.path.join(OUT, "tab_detect.tex"), "w", encoding="ascii") as t:
            t.write("\\begin{tabular}{lcccc}\n\\toprule\n & & \\multicolumn{2}{c}{mean recall} & weakest family \\\\\n\\cmidrule(lr){3-4}\nDetector & AUROC & @0.1\\%FPR & @1\\%FPR & @1\\%FPR \\\\\n\\midrule\n")
            names = {"xgb": "Gradient-boosted trees (XGBoost)", "rf": "Random forest", "mlp": "Multilayer perceptron", "iso": "Isolation forest (benign-only)"}
            for k in ("xgb", "rf", "mlp", "iso"):
                m = det["models"].get(k)
                if not m:
                    continue
                r01 = m.get("recall_fpr0.001", {}); r1 = m.get("recall_fpr0.01", {})
                wk = min(r1, key=r1.get) if r1 else "--"
                t.write("%s & %.3f & %.3f & %.3f & %s (%.2f) \\\\\n" % (names[k], m["auc"] or 0, np.mean(list(r01.values())), np.mean(list(r1.values())), NICE.get(wk, wk), r1.get(wk, 0)))
            t.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, "tab_knobs.tex"), "w", encoding="ascii") as t:
        t.write("\\begin{tabular}{llp{7.6cm}}\n\\toprule\nKnob & Range & What the attacker sets on its own host \\\\\n\\midrule\n")
        for n, r, dsc in KNOB_DESC:
            t.write("\\texttt{%s} & %s & %s \\\\\n" % (n, r, dsc))
        t.write("\\bottomrule\n\\end{tabular}\n")
    if atk and atk.get("families") and fail:
        fams = atk["families"]; FF = fail["families"]
        # run-level counts from the executed log (same source as the funcXXX macros and the Results text),
        # so the table's "runs/functional" columns match the prose and sum to nFunctionalRuns.
        runcounts = collections.defaultdict(lambda: [0, 0])
        for r in runs:
            if r.get("kind") == "attack":
                runcounts[r["family"]][0] += 1
                runcounts[r["family"]][1] += int(r.get("func_ok", 0))
        with open(os.path.join(OUT, "tab_attack.tex"), "w", encoding="ascii") as t:
            t.write("\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n & & & \\multicolumn{3}{c}{feature-space evasion (\\%)} & \\multicolumn{3}{c}{realised evasion (\\%)} & & \\\\\n")
            t.write("\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\nFamily & runs & functional & line & HSJA & masked & random & transfer & query & gap (pp) & closest margin \\\\\n\\midrule\n")
            for f in FAMS:
                if f not in fams:
                    continue
                v = fams[f]; F = FF.get(f, {})
                ntot, nfunc = runcounts.get(f, [0, 0])
                fr = "%d/%d" % (nfunc, ntot)
                mg = F.get("functional_logodds_margin", {}).get("min")
                t.write("%s & %d & %s & %.0f & %.0f & %.1f & %.0f & %.0f & %.0f & %.0f & %s \\\\\n" % (
                    NICE[f], ntot, fr, 100 * v["asr_featurespace_unconstrained"], 100 * (v.get("asr_featurespace_hsja") or 0),
                    100 * v["asr_featurespace_masked"], 100 * v["realised_random_knob"], 100 * v["realised_surrogate_transfer"],
                    100 * (v.get("realised_query_feedback") or 0), 100 * v["gap_fs_minus_realised"], ("%+.1f" % mg) if mg is not None else "--"))
            t.write("\\bottomrule\n\\end{tabular}\n")
        with open(os.path.join(OUT, "tab_nearmiss.tex"), "w", encoding="ascii") as t:
            t.write("\\begin{tabular}{llrrp{6.9cm}}\n\\toprule\nFamily & functionality (metric) & lowest score & margin & knob configuration of the closest functional run \\\\\n\\midrule\n")
            for f in FAMS:
                F = FF.get(f)
                if not F or not F.get("near_misses"):
                    continue
                r = F["near_misses"][0]; c = r["cfg"]; thr = logit(fail["threshold_fpr1"])
                mg = logit(r["min_score"]) - thr
                fm = r["func_metric"]; name = F["func_metric_name"]
                fms = ("%.0f$\\times$" % fm) if f == "coapamp" else ("%.2f" % fm)
                sp = ["fixed", "sequential", "random-high", "random-any"][int(c.get("srcport_mode", 0))]
                frag = ["none", "1500", "576", "256"][int(c.get("frag_mtu", 0))]
                cfg = ("rate %.0f pps, jitter %.2f, pad %d B, %d pkts/conn, parallel %d, src-port %s, TTL %d, window %.0f, MTU %s, delay %.0f ms, loss %.1f\\%%, idle %.1f s, cover %.2f"
                       % (c.get("rate_pps", 0), c.get("ipd_jitter", 0), c.get("pad_bytes", 0), c.get("pkts_per_conn", 0), c.get("parallel", 0), sp, c.get("ttl", 0),
                          c.get("tcp_window", 0), frag, c.get("netem_delay", 0), c.get("netem_loss", 0), c.get("idle_insert", 0), c.get("cover", 0)))
                t.write("%s & %s (%s) & %.3f & %+.1f & \\footnotesize %s \\\\\n" % (NICE[f], fms, name, r["min_score"], mg, cfg))
            t.write("\\bottomrule\n\\end{tabular}\n")
    print("b_report done ->", OUT)


if __name__ == "__main__":
    main()
