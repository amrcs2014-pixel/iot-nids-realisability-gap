"""Paper B (revision) - number macros, tables and figures from results/b_revision.json into paperB/.

Figures
  b_fig_pipeline   methodology diagram (twin -> library + sweep -> flows -> detectors (2 regimes) -> two arms -> gap)
  b_fig_regimes    realised evasion and silent functional runs, every detector x training regime x family
  b_fig_gap        feature-space vs realised evasion for XGBoost (natural vs knob-augmented); L2 in z and raw units
  b_fig_sweep      targeted one-factor-at-a-time sweep: functionality and evasion per knob level
  b_fig_mechanism  displacement the feature-space attack needs, by feature group; detector importances
  b_fig_ablation   realised evasion after dropping / quantising the dominant features; dst2src_bytes vs benign range
  b_fig_perturb    analytic perturb-ability score vs executed reachability, per feature
  b_fig_gotham     benign realism (KS) and transfer in both directions, with threshold re-calibration
Tables
  tab_knobs, tab_knobmap, tab_library, tab_detect, tab_realised, tab_fs, tab_ablation, tab_gotham, tab_perturb (appendix)
"""
import json, os, collections, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_report import box, arrow, C, NICE, FAMS, group_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "paperB")
DATA = os.path.join(ROOT, "data", os.environ.get("BDATA", "btwin"))
RJ = os.environ.get("BREV", "b_revision.json")
plt.rcParams.update({"font.size": 8.5, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9, "legend.fontsize": 7.5})
DET = [("xgb", "natural"), ("xgb", "knob"), ("rf", "natural"), ("rf", "knob"), ("mlp", "natural"), ("mlp", "knob"), ("iso", "benign_only")]
DNAME = {"xgb": "XGBoost", "rf": "Random forest", "mlp": "MLP", "iso": "Isolation forest"}
RNAME = {"natural": "natural", "knob": "knob-aug.", "benign_only": "benign-only"}
KNOBS = ["rate_pps", "ipd_jitter", "pad_bytes", "pkts_per_conn", "parallel", "srcport_mode", "ttl", "tcp_window",
         "frag_mtu", "netem_delay", "netem_loss", "idle_insert"]
KNOB_DESC = {  # range, natural value, what the attacker sets
    "rate_pps": ("1--400 (log)", "200", "offered send rate (packets/s; per sender for flooding)"),
    "ipd_jitter": ("0--1", "0", "uniform jitter as a fraction of the mean inter-packet delay"),
    "pad_bytes": ("0--512", "0", "payload padding appended to every packet or request"),
    "pkts_per_conn": ("1--40", "1", "credential attempts per TCP connection (brute-forcing; the service answers at most six)"),
    "parallel": ("1--8", "1", "concurrent flooding senders"),
    "srcport_mode": ("fixed / sequential / random-high / random-any", "random-high", "source-port strategy"),
    "ttl": ("32--128", "64", "IP time-to-live stamped on outgoing packets"),
    "tcp_window": ("512--65535 (log)", "29200", "advertised TCP window of crafted SYNs"),
    "frag_mtu": ("none / 1500 / 576 / 256", "none", "IP fragmentation of crafted packets"),
    "netem_delay": ("0--100 ms", "0", "egress delay on the attacker NIC (\\texttt{tc netem})"),
    "netem_loss": ("0--5\\%", "0", "egress loss on the attacker NIC (\\texttt{tc netem})"),
    "idle_insert": ("0--3 s", "0", "idle gap inserted after every burst"),
}
# how each implemented knob reaches the nfstream feature vector (see b_hosts.py and the Methodology)
KMAP = {
    "netscan": {"rate_pps": "X", "ipd_jitter": "X", "pad_bytes": "F", "srcport_mode": "X", "ttl": "0", "tcp_window": "0",
                "idle_insert": "X", "netem_delay": "F", "netem_loss": "F"},
    "svcbrute": {"rate_pps": "F", "ipd_jitter": "F", "pad_bytes": "F", "pkts_per_conn": "F", "idle_insert": "X",
                 "netem_delay": "F", "netem_loss": "F"},
    "synflood": {"rate_pps": "F*", "ipd_jitter": "F*", "pad_bytes": "F", "parallel": "F*", "srcport_mode": "S", "ttl": "0",
                 "tcp_window": "0", "frag_mtu": "F", "idle_insert": "F*", "netem_delay": "F", "netem_loss": "F"},
    "coapamp": {"rate_pps": "F*", "ipd_jitter": "F*", "srcport_mode": "S", "ttl": "0", "idle_insert": "F*",
                "netem_delay": "F", "netem_loss": "F"},
    "httpbeacon": {"rate_pps": "X", "ipd_jitter": "X", "pad_bytes": "F", "netem_delay": "F", "netem_loss": "F"},
}


def tex(s):
    return str(s).replace("_", "\\_")


def pct(x, nd=1):
    return "--" if x is None else f"{100 * x:.{nd}f}"


def main():
    R = json.load(open(os.path.join(RES, RJ)))
    os.makedirs(OUT, exist_ok=True)
    M = collections.OrderedDict()

    def macro(k, v, nd=3):
        M[k] = f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    R2 = R["R2_detectors"]; R3 = R["R3_feature_space"]; R4 = R.get("R4_ablation", {}); R6 = R.get("R6_gotham", {})
    sh = R["R1_shape"]

    # ---------------- library counts (from the run log) ----------------
    runs = []
    rp = os.path.join(DATA, "b_runs.jsonl")
    if os.path.exists(rp):
        for l in open(rp, encoding="utf-8"):
            try:
                runs.append(json.loads(l))
            except Exception:
                pass
    atk = [r for r in runs if r.get("kind") == "attack"]
    macro("nRuns", len(runs)); macro("nBenignRuns", sum(r.get("kind") == "benign" for r in runs)); macro("nAttackRuns", len(atk))
    macro("nCfg", 120); macro("nKnobs", len(KNOBS)); macro("slowBudget", "30"); macro("hsjaCap", "3{,}000")
    macro("nFeatures", R["n_features"]); macro("nFlows", f"{R['n_flows']:,}".replace(",", "{,}"))
    macro("nAttackFlows", f"{R['n_attack_flows']:,}".replace(",", "{,}")); macro("nCalBenign", f"{R['n_benign_flows']:,}".replace(",", "{,}"))
    macro("nCalRuns", R["n_benign_runs"])
    macro("coapAmp", f"{R['R1_coap_amplification']['median']:.1f}")
    nfunc = {f: sh[f]["func_natural"][0] + sh[f]["func_sobol"][0] for f in FAMS}
    macro("nFunctionalRuns", sum(nfunc.values()))
    for f in FAMS:
        macro("funcNat" + f, f"{sh[f]['func_natural'][0]}/{sh[f]['func_natural'][1]}")
        macro("funcSob" + f, f"{sh[f]['func_sobol'][0]}/{sh[f]['func_sobol'][1]}")
    sw = R.get("R8_sweep", {})
    macro("nSweepRuns", sw.get("n_runs", "--"))

    # ---------------- detectors ----------------
    def r2(k):
        return R2.get(k, {})
    xn, xk, iso = r2("xgb:natural"), r2("xgb:knob"), r2("iso:benign_only")
    macro("aucXgbKnob", xk.get("auc_oof", 0.0)); macro("aucXgbNat", xn.get("auc_oof", 0.0)); macro("aucIso", iso.get("auc_oof", 0.0))
    for key, nm in (("xgb:natural", "XgbNat"), ("xgb:knob", "XgbKnob"), ("rf:natural", "RfNat"), ("rf:knob", "RfKnob"),
                    ("mlp:natural", "MlpNat"), ("mlp:knob", "MlpKnob"), ("iso:benign_only", "Iso")):
        d = r2(key)
        if not d:
            continue
        rr = d["realised"]
        rr_f = {f: rr[f].get("r_rand") for f in FAMS if rr.get(f, {}).get("n_func_flows")}
        for f, v in rr_f.items():
            macro("rr" + nm + f, pct(v))
        sil = sum(rr[f].get("runs_no_alert", 0) for f in FAMS if rr.get(f)); tot = sum(rr[f].get("n_func_runs", 0) for f in FAMS if rr.get(f))
        macro("silent" + nm, f"{sil}/{tot}")
        macro("benRunAlarm" + nm, pct(d.get("benign_run_alarm_rate"), 0))
        cr = [v for v in d["clean_recall_natural"].values() if v is not None]
        macro("clean" + nm, pct(np.mean(cr)))
        macro("rrMax" + nm, pct(max(rr_f.values())) if rr_f else "--")
        macro("rqMax" + nm, pct(max(rr[f].get("r_query") or 0 for f in rr_f)) if rr_f else "--")
    # feature-space headline: mean line-search ASR over families and detectors
    asr = [v.get("asr_line") for d in R3.values() for v in d.values() if v.get("asr_line") is not None]
    macro("asrFsMean", pct(np.mean(asr)) if asr else "--")
    hs = [v.get("asr_hsja") for d in R3.values() for v in d.values() if v.get("asr_hsja") is not None]
    macro("asrHsjaMean", pct(np.mean(hs)) if hs else "--")
    msk = [v.get("asr_masked") for d in R3.values() for v in d.values() if v.get("asr_masked") is not None]
    macro("asrMaskedMean", pct(np.mean(msk)) if msk else "--")
    feas = collections.defaultdict(list)
    for d in R3.values():
        for v in d.values():
            for k, val in v.get("feasibility", {}).items():
                feas[k].append(val)
    for k in ("infeasible", "needs_noncontrollable", "controllable_only"):
        macro("feas" + k.replace("_", ""), pct(np.mean(feas[k])) if feas[k] else "--")

    # ---------------- ablation / controllability ----------------
    for regime in ("knob", "natural"):
        top = R.get(f"R4_full_top_features_{regime}", [])
        if top:
            macro("topFeat" + regime, tex(top[0][0])); macro("topImp" + regime, top[0][1], 2)
            macro("topThree" + regime, pct(R.get(f"R4_full_top3_gain_share_{regime}"), 0))
    # ---------------- perturb-ability ----------------
    P = R.get("R7_perturbability", {})
    if P:
        macro("pertSpearman", P["spearman_ps_vs_reach"], 2)
        cc = P["class_counts"]
        macro("psLow", cc.get("low", 0)); macro("psMed", cc.get("medium", 0)); macro("psHigh", cc.get("high", 0))
        for k, v in P.get("mean_reach_by_class", {}).items():
            macro("reach" + k, v, 2)
    # ---------------- Gotham ----------------
    if R6:
        a = R6["twin_detector_on_gotham"]; b = R6["gotham_detector_on_twin"]; br = R6.get("benign_realism", {})
        macro("nGotham", f"{R6['n_gotham_flows']:,}".replace(",", "{,}")); macro("nGothamBenign", f"{R6['n_gotham_benign']:,}".replace(",", "{,}"))
        macro("gothAuc", a["auc"]); macro("gothRecallTwinThr", pct(a["at_twin_threshold"]["recall_real_attack"]))
        macro("gothFprTwinThr", pct(a["at_twin_threshold"]["fpr_real_benign"]))
        rc = a["recalibrated_1pct_on_real_benign"]
        macro("gothRecallCal", pct(rc["recall_real_attack"])); macro("gothFprCal", pct(rc["fpr_real_benign_heldout"]))
        macro("gothToTwinAuc", b["auc_on_twin"]); macro("gothToTwinFpr", pct(b["fpr_twin_benign"]))
        macro("gothAucOof", b["gotham_auc_oof"])
        nr = [v for v in b["recall_twin_natural"].values()]
        macro("gothToTwinRecall", pct(np.mean(nr)))
        if br:
            macro("ksBefore", br["ks_median"].get("before", float("nan")), 2); macro("ksMed", br["ks_median"].get("after", float("nan")), 2)
            macro("discAuc", br["discriminator_auc"])
    with open(os.path.join(OUT, "numbers_B.tex"), "w", encoding="ascii") as fh:
        for k, v in M.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    write_tables(R, runs)
    fig_pipeline(M)
    fig_regimes(R)
    fig_gap(R)
    if sw:
        fig_sweep(R)
    fig_mechanism(R)
    fig_ablation(R)
    if P:
        fig_perturb(R)
    if R6:
        fig_gotham(R)
    print("b_report2 done ->", OUT, len(M), "macros")


# =============================================================== tables
def write_tables(R, runs):
    sh = R["R1_shape"]; R2 = R["R2_detectors"]; R3 = R["R3_feature_space"]
    with open(os.path.join(OUT, "tab_knobs.tex"), "w", encoding="ascii") as t:
        t.write("\\begin{tabular}{p{2.3cm}p{3.0cm}lp{6.2cm}}\n\\toprule\nKnob & Range & natural & What the attacker sets on its own host \\\\\n\\midrule\n")
        for k in KNOBS:
            rg, nat, dsc = KNOB_DESC[k]
            t.write("\\texttt{%s} & %s & \\textbf{%s} & %s \\\\\n" % (tex(k), rg, nat, dsc))
        t.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, "tab_knobmap.tex"), "w", encoding="ascii") as t:
        t.write("\\begin{tabular}{l" + "c" * len(FAMS) + "}\n\\toprule\nKnob & " + " & ".join(NICE[f] for f in FAMS) + " \\\\\n\\midrule\n")
        for k in KNOBS:
            cells = []
            for f in FAMS:
                v = KMAP[f].get(k, "--")
                cells.append({"F*": "F$^\\ast$"}.get(v, "\\textbf{%s}" % v if v in ("F", "S") else v))
            t.write("\\texttt{%s} & %s \\\\\n" % (tex(k), " & ".join(cells)))
        t.write("\\midrule\nflows per run (median) & " + " & ".join("%.0f" % sh[f]["flows_per_run_median"] for f in FAMS) + " \\\\\n")
        t.write("packets per flow (median) & " + " & ".join("%.0f" % sh[f]["pkts_per_flow_median"] for f in FAMS) + " \\\\\n")
        t.write("flow duration, ms (median) & " + " & ".join("%.0f" % sh[f]["duration_ms_median"] for f in FAMS) + " \\\\\n")
        t.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, "tab_library.tex"), "w", encoding="ascii") as t:
        budget = {"netscan": "30 s (ends when done)", "svcbrute": "30 s (ends when done)", "synflood": "6 s", "coapamp": "6 s", "httpbeacon": "6 s"}
        t.write("\\begin{tabular}{lllrrr}\n\\toprule\n & & & \\multicolumn{2}{c}{functional runs} & attack \\\\\n\\cmidrule(lr){4-5}\n"
                "Family & functionality oracle & budget & natural & Sobol & flows \\\\\n\\midrule\n")
        orc = {"netscan": "both open ports found", "svcbrute": "credential accepted", "synflood": "goodput drop $\\geq$20\\%",
               "coapamp": "requests reflected", "httpbeacon": "$\\geq$80\\% beacons acked"}
        for f in FAMS:
            t.write("%s & %s & %s & %d/%d & %d/%d & %s \\\\\n" % (NICE[f], orc[f], budget[f], sh[f]["func_natural"][0], sh[f]["func_natural"][1],
                                                               sh[f]["func_sobol"][0], sh[f]["func_sobol"][1], f"{sh[f]['flows']:,}".replace(",", "{,}")))
        t.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, "tab_detect.tex"), "w", encoding="ascii") as t:
        t.write("\\begin{tabular}{llccccc}\n\\toprule\n & & & \\multicolumn{2}{c}{clean recall, natural runs} & \\multicolumn{2}{c}{benign false alarms} \\\\\n"
                "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nDetector & training & AUROC & mean & weakest family & flows & captures \\\\\n\\midrule\n")
        for d, rg in DET:
            v = R2.get(f"{d}:{rg}")
            if not v:
                continue
            cr = {f: x for f, x in v["clean_recall_natural"].items() if x is not None}
            wk = min(cr, key=cr.get)
            t.write("%s & %s & %.3f & %s & %s (%s) & %s & %s \\\\\n" % (DNAME[d], RNAME[rg], v["auc_oof"], pct(np.mean(list(cr.values()))), NICE[wk],
                                                                    pct(cr[wk], 0), pct(v["fpr_oof"]), pct(v.get("benign_run_alarm_rate"), 0)))
        t.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, "tab_realised.tex"), "w", encoding="ascii") as t:
        t.write("\\begin{tabular}{ll" + "c" * len(FAMS) + "c}\n\\toprule\n & & \\multicolumn{%d}{c}{realised flow evasion: random / transfer / query (\\%%)} & silent \\\\\n" % len(FAMS))
        t.write("\\cmidrule(lr){3-%d}\nDetector & training & %s & runs \\\\\n\\midrule\n" % (2 + len(FAMS), " & ".join(NICE[f] for f in FAMS)))
        for d, rg in DET:
            v = R2.get(f"{d}:{rg}")
            if not v:
                continue
            cells = []; sil = tot = 0
            for f in FAMS:
                x = v["realised"].get(f, {})
                if not x.get("n_func_flows"):
                    cells.append("--"); continue
                cells.append("%s\\,/\\,%s\\,/\\,%s" % (pct(x["r_rand"], 0), pct(x.get("r_transfer"), 0), pct(x["r_query"], 0)))
                sil += x["runs_no_alert"]; tot += x["n_func_runs"]
            t.write("%s & %s & %s & %d/%d \\\\\n" % (DNAME[d], RNAME[rg], " & ".join(cells), sil, tot))
        t.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, "tab_fs.tex"), "w", encoding="ascii") as t:
        t.write("\\begin{tabular}{lll" + "c" * 7 + "}\n\\toprule\n & & & \\multicolumn{3}{c}{feature-space ASR (\\%)} & \\multicolumn{2}{c}{median $L_2$, line search} & & \\\\\n")
        t.write("\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\nDetector & training & family & line & HSJA & masked & standardised & raw & labels I/N/C (\\%) & $\\Delta$ (pp) \\\\\n\\midrule\n")
        for d, rg in DET:
            key = f"{d}:{rg}"; v = R3.get(key); w = R2.get(key)
            if not v:
                continue
            first = True
            for f in FAMS:
                x = v.get(f, {})
                if "asr_line" not in x:
                    continue
                rt = w["realised"].get(f, {}).get("r_transfer")
                gap = None if rt is None else 100 * (x["asr_line"] - rt)
                fe = x["feasibility"]
                t.write("%s & %s & %s & %s & %s & %s & %s & %s & %s/%s/%s & %s \\\\\n" % (
                    DNAME[d] if first else "", RNAME[rg] if first else "", NICE[f], pct(x["asr_line"], 0), pct(x.get("asr_hsja"), 0),
                    pct(x["asr_masked"], 0), "--" if x.get("l2_z_line_med") is None else "%.1f" % x["l2_z_line_med"],
                    "--" if x.get("l2_raw_line_med") is None else "%.0f" % x["l2_raw_line_med"],
                    pct(fe.get("infeasible"), 0), pct(fe.get("needs_noncontrollable"), 0), pct(fe.get("controllable_only"), 0),
                    "--" if gap is None else "%.0f" % gap))
                first = False
            if (d, rg) != DET[-1]:
                t.write("\\addlinespace\n")
        t.write("\\bottomrule\n\\end{tabular}\n")
    R4 = R.get("R4_ablation", {})
    if R4:
        with open(os.path.join(OUT, "tab_ablation.tex"), "w", encoding="ascii") as t:
            names = {"drop_top2": "drop \\texttt{dst2src\\_bytes}, \\texttt{bidirectional\\_min\\_ps}", "drop_victim_side": "drop all victim-side (\\texttt{dst2src\\_*})",
                     "no_size_features": "drop all byte and packet-size features", "log2_quantised": "all features $\\log_2$-quantised"}
            t.write("\\begin{tabular}{llcc" + "c" * len(FAMS) + "}\n\\toprule\n & & & clean & \\multicolumn{%d}{c}{realised flow evasion, random knob (\\%%)} \\\\\n" % len(FAMS))
            t.write("\\cmidrule(lr){5-%d}\nFeature set & training & AUROC & recall & %s \\\\\n\\midrule\n" % (4 + len(FAMS), " & ".join(NICE[f] for f in FAMS)))
            for regime in ("natural", "knob"):
                full = R2[f"xgb:{regime}"]
                rows = [("all %d features" % R["n_features"], full)] + [(names[n], R4[f"xgb:{regime}:{n}"]) for n in names if f"xgb:{regime}:{n}" in R4]
                for i, (nm, v) in enumerate(rows):
                    cr = [x for x in v["clean_recall_natural"].values() if x is not None]
                    t.write("%s & %s & %.3f & %s & %s \\\\\n" % (nm, RNAME[regime] if i == 0 else "", v["auc_oof"], pct(np.mean(cr)),
                                                            " & ".join(pct(v["realised"].get(f, {}).get("r_rand"), 0) for f in FAMS)))
                if regime == "natural":
                    t.write("\\addlinespace\n")
            t.write("\\bottomrule\n\\end{tabular}\n")
    R6 = R.get("R6_gotham", {})
    if R6:
        a = R6["twin_detector_on_gotham"]; b = R6["gotham_detector_on_twin"]
        with open(os.path.join(OUT, "tab_gotham.tex"), "w", encoding="ascii") as t:
            labs = {"coapamp": "coap-amp", "merlin_c2": "HTTP C2", "mirai_infect": "Mirai infection", "netscan": "scan", "synflood": "flood"}
            t.write("\\begin{tabular}{lcc}\n\\toprule\n\\multicolumn{3}{l}{\\textit{(i) twin-trained XGBoost applied to real Gotham traffic (AUROC %.3f)}} \\\\\n" % a["auc"])
            t.write(" & twin threshold & re-calibrated, 1\\% real FPR \\\\\n\\midrule\n")
            t.write("false-positive rate on real benign (\\%%) & %s & %s \\\\\n" % (pct(a["at_twin_threshold"]["fpr_real_benign"]), pct(a["recalibrated_1pct_on_real_benign"]["fpr_real_benign_heldout"])))
            t.write("recall on real attacks, all (\\%%) & %s & %s \\\\\n" % (pct(a["at_twin_threshold"]["recall_real_attack"]), pct(a["recalibrated_1pct_on_real_benign"]["recall_real_attack"])))
            for k, v in a["recalibrated_1pct_on_real_benign"]["recall_by_family"].items():
                t.write("\\quad %s (\\%%) & & %s \\\\\n" % (labs.get(k, k), pct(v)))
            t.write("\\midrule\n\\multicolumn{3}{l}{\\textit{(ii) Gotham-trained XGBoost applied to the twin (threshold: 1\\% FPR on real benign)}} \\\\\n")
            t.write("AUROC on twin flows & \\multicolumn{2}{c}{%.3f} \\\\\n" % b["auc_on_twin"])
            t.write("false-positive rate on twin benign (\\%%) & \\multicolumn{2}{c}{%s} \\\\\n" % pct(b["fpr_twin_benign"]))
            t.write("recall on natural twin attacks, mean (\\%%) & \\multicolumn{2}{c}{%s} \\\\\n" % pct(np.mean(list(b["recall_twin_natural"].values()))))
            rr = b["realised"]
            t.write("realised flow evasion, random knob (\\%%) & \\multicolumn{2}{c}{%s} \\\\\n" % ", ".join("%s %s" % (NICE[f], pct(rr[f].get("r_rand"), 0)) for f in FAMS if rr.get(f, {}).get("n_func_flows")))
            t.write("silent functional runs & \\multicolumn{2}{c}{%d/%d} \\\\\n" % (sum(rr[f].get("runs_no_alert", 0) for f in FAMS), sum(rr[f].get("n_func_runs", 0) for f in FAMS)))
            t.write("\\bottomrule\n\\end{tabular}\n")
    P = R.get("R7_perturbability", {})
    if P:
        pf = P["per_feature"]
        feats = sorted(pf, key=lambda c: (-pf[c]["PS"], -pf[c]["executed_reach"]))
        with open(os.path.join(OUT, "tab_perturb.tex"), "w", encoding="ascii") as t:
            t.write("\\begin{longtable}{lccccccc}\n\\caption{\\textbf{Perturb-ability score of the 59 nfstream features.} Sub-scores PS1--PS5 and the total PS as defined by ElShehaby and Matrawy, the resulting class, and the executed reachability (mean over families of the largest $|\\rho_s|$ between an implemented knob and the feature on functional Sobol flows).}\\label{tab:perturb}\\\\\n")
            t.write("\\toprule\nFeature & PS1 & PS2 & PS3 & PS4 & PS5 & PS (class) & reach \\\\\n\\midrule\n\\endfirsthead\n\\toprule\nFeature & PS1 & PS2 & PS3 & PS4 & PS5 & PS (class) & reach \\\\\n\\midrule\n\\endhead\n")
            for c in feats:
                v = pf[c]
                t.write("\\texttt{\\footnotesize %s} & %.0f & %.2f & %.2f & %.1f & %.1f & %.2f (%s) & %.2f \\\\\n" % (tex(c), v["PS1"], v["PS2"], v["PS3"], v["PS4"], v["PS5"], v["PS"], v["class"], v["executed_reach"]))
            t.write("\\bottomrule\n\\end{longtable}\n")


# =============================================================== figures
def fig_pipeline(M):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    def n(k):
        return str(M.get(k, "")).replace("{,}", ",")
    # edge, face per legend category
    COL = {"real": ("#D1912A", "#FBF2DC"), "emul": ("#2E9E8F", "#E3F3F0"), "knob": ("#CE7FA8", "#F8E7F1"),
           "det": ("#7D7D7D", "#EFEFEF"), "fs": ("#D9702E", "#FBE7D9"), "arm": ("#4587BE", "#E2EEF7"),
           "gap": ("#1A1A1A", "#FFFFFF")}
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(-5, 64); ax.axis("off")

    def badge(x, y, num, key):
        ax.add_patch(Circle((x + 2.4, y), 2.15, fc=COL[key][0], ec="white", lw=1.1, zorder=6))
        ax.text(x + 2.4, y - 0.05, str(num), ha="center", va="center", color="white", fontsize=6.6, fontweight="bold", zorder=7)

    def nbox(x, y, w, h, num, key, title, lines, tfs=7.2, bfs=5.5, dashed=False):
        e, f = COL[key]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3",
                                    fc=f, ec=e, lw=1.3, ls=(0, (4, 2)) if dashed else "-", zorder=2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=tfs, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h - 2.6 - tfs * 0.52, lines, ha="center", va="top", fontsize=bfs, linespacing=1.5, zorder=4)
        badge(x, y + h, num, key)

    def arr(p, q, color="#555555", dashed=False, lw=1.1):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw,
                                     ls=(0, (4, 2)) if dashed else "-", shrinkA=1, shrinkB=1, zorder=1))

    # ---- row 1
    y1, h1 = 50, 12.5
    nbox(1, y1, 21, h1, 1, "real", "Reference traffic", f"real GothamDataset2025 pcaps\nbenign + attack captures\n{n('nGotham')} flows ({n('nGothamBenign')} benign)")
    nbox(25, y1, 24.5, h1, 2, "emul", "IoT digital twin", "Containernet + Open vSwitch\nMQTT · CoAP · HTTP · TCP login\n4 calibrated devices + attacker\n10 Mbit/s shaped victim link", bfs=5.3)
    nbox(52.5, y1, 23, h1, 3, "knob", "Attacker knob catalogue", f"{n('nKnobs')} host-level knobs\n5 sanitised attack families\nSobol design: 120 configs/family\n+ 12 natural repeats per family\none-factor-at-a-time sweep", bfs=5.2)
    nbox(78.5, y1, 20.5, h1, 4, "emul", "Executed knob library", f"{n('nRuns')} runs ({n('nAttackRuns')} attack + {n('nBenignRuns')} benign)\n+ {n('nSweepRuns')} sweep runs (eval. only)\npcap + knob vector per run\nfunctionality oracle + cost", bfs=5.1)
    arr((22, y1 + h1 / 2), (25, y1 + h1 / 2), color=COL["real"][0])
    arr((49.5, y1 + h1 / 2), (52.5, y1 + h1 / 2)); arr((75.5, y1 + h1 / 2), (78.5, y1 + h1 / 2))

    # ---- row 2
    y2, h2 = 27.5, 17
    nbox(1, y2, 21, h2, 10, "real", "External validity", "twin $\\rightarrow$ real Gotham\nreal Gotham $\\rightarrow$ twin\nper-feature KS distance\ntwin-vs-real discriminator\nthreshold re-calibration", bfs=5.3)
    nbox(78.5, y2, 20.5, h2, 5, "det", "Flow extraction", f"nfstream, bidirectional\n5-tuple flows\nidle 15 s / active 120 s\n{n('nFeatures')} statistical features\nidentifiers dropped", bfs=5.3)
    # box 6: detectors with three sub-boxes
    e6, f6 = COL["det"]
    ax.add_patch(FancyBboxPatch((25, y2), 49, h2, boxstyle="round,pad=0,rounding_size=1.3", fc=f6, ec=e6, lw=1.3, zorder=2))
    ax.text(42, y2 + h2 - 2.4, "Flow detectors", ha="center", va="top", fontsize=7.4, fontweight="bold", zorder=4)
    ax.text(62.6, y2 + h2 - 2.9, "(cross-fitted by run and configuration)", ha="center", va="top", fontsize=5.4, style="italic", zorder=4)
    badge(25, y2 + h2, 6, "det")
    subs = [("Natural-only", "XGBoost · RF · MLP\nattack class = the 12\ndefault runs per family", False),
            ("Knob-augmented", "XGBoost · RF · MLP\n+ Sobol knob\nconfigurations", False),
            ("Benign-only", "isolation forest\nzero-day setting:\nno attack seen", True)]
    sx = 26.5
    for t, b, dash in subs:
        sw = 15.0
        ax.add_patch(FancyBboxPatch((sx, y2 + 4.2), sw, 8.2, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="white", ec="#555555", lw=0.9, ls=(0, (3, 2)) if dash else "-", zorder=3))
        ax.text(sx + sw / 2, y2 + 11.4, t, ha="center", va="top", fontsize=5.8, fontweight="bold", zorder=4)
        ax.text(sx + sw / 2, y2 + 9.4, b, ha="center", va="top", fontsize=4.9, linespacing=1.35, zorder=4)
        sx += sw + 1.25
    ax.text(49.5, y2 + 2.3, "operating point fixed once: $\\tau$ = 1% out-of-fold benign FPR, unchanged in every arm",
            ha="center", va="center", fontsize=5.2, zorder=4)
    arr((95, y1), (95, y2 + h2), color=COL["emul"][0]); ax.text(96, (y1 + y2 + h2) / 2, "pcaps", fontsize=5.0, style="italic", color=COL["emul"][0], ha="left", va="center")
    arr((78.5, y2 + h2 / 2), (74, y2 + h2 / 2))
    arr((25, y2 + h2 / 2), (22, y2 + h2 / 2), dashed=True, color=COL["real"][0])
    arr((11.5, y1), (11.5, y2 + h2), dashed=True, color=COL["real"][0]); ax.text(12.4, (y1 + y2 + h2) / 2, "real flows", fontsize=5.0, style="italic", color=COL["real"][0], ha="left", va="center")

    # ---- row 3
    y3, h3 = 1.5, 18
    nbox(1, y3, 28, h3, 7, "fs", "Feature-space arm", "boundary line search\nHopSkipJump (3,000 queries)\ncontrollable-only mask\nin $z$-standardised space\n$\\rightarrow \\mathrm{ASR_{fs}}$, $L_2$, feasibility labels", bfs=5.3)
    nbox(70, y3, 29, h3, 8, "arm", "Realised arm", "functional flows only\nrandom knob $r_{\\mathrm{rand}}$\nsurrogate transfer $r_{\\mathrm{transfer}}$\nquery-feedback bound $r_{\\mathrm{query}}$\n+ silent functional runs", bfs=5.3)
    # box 9: realisability gap (equation)
    e9, f9 = COL["gap"]
    ax.add_patch(FancyBboxPatch((34, y3 + 0.5), 30, h3 - 1, boxstyle="round,pad=0,rounding_size=1.3", fc=f9, ec=e9, lw=1.7, zorder=3))
    ax.text(49, y3 + h3 - 2.4, "Realisability gap", ha="center", va="top", fontsize=8.0, fontweight="bold", zorder=4)
    ax.text(49, y3 + h3 - 7.3, "$\\Delta = \\mathrm{ASR_{fs}} - r_{\\mathrm{transfer}}$", ha="center", va="center", fontsize=11, zorder=4)
    ax.text(49, y3 + h3 - 10.2, "per family × detector × training regime", ha="center", va="center", fontsize=5.3, zorder=4)
    ax.text(49, y3 + 2.7, "explained by: displacement per feature group,\nfeature ablations, perturb-ability vs. reachability",
            ha="center", va="center", fontsize=4.7, style="italic", color="#444444", linespacing=1.4, zorder=4)
    badge(34, y3 + h3, 9, "gap")
    # detectors -> arms, and functional flows -> realised arm
    arr((40, y2), (15, y3 + h3), color="#555555")
    ax.text(29, 22.0, "detector $D$, threshold $\\tau$", fontsize=5.2, style="italic", color="#555555", ha="center")
    arr((84, y2), (84, y3 + h3), color=COL["emul"][0])
    ax.text(85, 22.0, "functional flows", fontsize=5.2, style="italic", color=COL["emul"][0], ha="left")
    arr((29, y3 + h3 / 2), (34, y3 + h3 / 2), color=COL["fs"][0], lw=1.4)
    arr((70, y3 + h3 / 2), (64, y3 + h3 / 2), color=COL["arm"][0], lw=1.4)

    # ---- legend
    leg = [("real", "real traffic"), ("emul", "emulated & executed"), ("knob", "attacker knobs"),
           ("det", "detectors"), ("fs", "feature space"), ("arm", "realised (problem space)")]
    lx = 2.0
    for key, lab in leg:
        ax.add_patch(FancyBboxPatch((lx, -3.4), 3.2, 2.0, boxstyle="round,pad=0,rounding_size=0.5", fc=COL[key][1], ec=COL[key][0], lw=1.1))
        ax.text(lx + 3.9, -2.4, lab, fontsize=5.2, va="center", ha="left")
        lx += 5.0 + len(lab) * 0.72
    ax.add_patch(FancyArrowPatch((lx, -2.4), (lx + 3.2, -2.4), arrowstyle="-|>", mutation_scale=9, color=COL["real"][0], lw=1.2, ls=(0, (4, 2))))
    ax.text(lx + 3.9, -2.4, "calibration / validation", fontsize=5.2, va="center", ha="left")

    fig.savefig(os.path.join(OUT, "b_fig_pipeline.pdf"), bbox_inches="tight"); plt.close(fig)


def heat(ax, M, rows, cols, fmt, cmap, vmax, title, annot_txt=None):
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.4)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.2, color="#777777"); continue
            s = annot_txt[i][j] if annot_txt else fmt % M[i, j]
            ax.text(j, i, s, ha="center", va="center", fontsize=6.6, color="white" if M[i, j] > 0.6 * vmax else "black")
    ax.set_title(title, loc="left", fontsize=8.3)
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    return im


def fig_regimes(R):
    R2 = R["R2_detectors"]
    rows = [f"{DNAME[d]} ({RNAME[rg]})" for d, rg in DET if f"{d}:{rg}" in R2]
    keys = [f"{d}:{rg}" for d, rg in DET if f"{d}:{rg}" in R2]
    A = np.full((len(keys), len(FAMS)), np.nan); B = A.copy(); Bt = [["" for _ in FAMS] for _ in keys]
    for i, k in enumerate(keys):
        for j, f in enumerate(FAMS):
            x = R2[k]["realised"].get(f, {})
            if x.get("n_func_flows"):
                A[i, j] = 100 * x["r_rand"]
                B[i, j] = 100 * x["runs_no_alert"] / max(1, x["n_func_runs"]); Bt[i][j] = f"{x['runs_no_alert']}/{x['n_func_runs']}"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.2), sharey=True)
    heat(a, A, rows, [NICE[f] for f in FAMS], "%.1f", "Reds", max(10, np.nanmax(A)), "(a) functional attack flows called benign (%)")
    heat(b, B, rows, [NICE[f] for f in FAMS], "%.0f", "Reds", max(10, np.nanmax(B)), "(b) silent functional runs (no flow flagged)", Bt)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_regimes.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_gap(R):
    R2 = R["R2_detectors"]; R3 = R["R3_feature_space"]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.3), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.27, wspace=0.38)
    x = np.arange(len(FAMS)); w = 0.13
    series = [("asr_line", "FS: line search", C["fs"]), ("asr_hsja", "FS: HopSkipJump", C["hsja"]), ("asr_masked", "FS: controllable only", C["masked"]),
              ("r_rand", "realised: random", C["rand"]), ("r_transfer", "realised: transfer", C["transfer"]), ("r_query", "realised: query bound", C["query"])]
    for ax, regime in zip(axs[:2], ("natural", "knob")):
        key = f"xgb:{regime}"
        for i, (k, lab, col) in enumerate(series):
            src = R3[key] if k.startswith("asr") else {f: R2[key]["realised"].get(f, {}) for f in FAMS}
            v = [100 * (src.get(f, {}).get(k) or 0.0) for f in FAMS]
            bars = ax.bar(x + (i - 2.5) * w, v, w, color=col, label=lab)
            for bx, vv in zip(bars, v):
                if vv == 0:
                    ax.plot([bx.get_x() + w / 2], [1.5], marker="v", ms=2.6, color=col, clip_on=False)
        ax.set_xticks(x); ax.set_xticklabels([NICE[f] for f in FAMS], rotation=30, ha="right"); ax.set_ylim(0, 108)
        ax.set_title(f"({'a' if regime == 'natural' else 'b'}) XGBoost, {RNAME[regime]} training", loc="left", fontsize=8.2)
    axs[0].set_ylabel("evasion at 1% benign FPR (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=6.4, handlelength=1.2, columnspacing=0.9)
    c = axs[2]
    for i, (regime, col) in enumerate((("natural", "#9ecae1"), ("knob", "#3182bd"))):
        v = [R3[f"xgb:{regime}"].get(f, {}).get("l2_z_line_med") or 0 for f in FAMS]
        c.bar(x + (i - 0.5) * 0.36, v, 0.36, color=col, label=RNAME[regime])
    c.set_yscale("log"); c.set_xticks(x); c.set_xticklabels([NICE[f] for f in FAMS], rotation=30, ha="right")
    lo_, hi_ = c.get_ylim(); c.set_ylim(lo_, hi_ * 8)          # head-room for the legend
    c.set_ylabel("median $L_2$ (standardised)"); c.legend(frameon=False, fontsize=6.4, loc="upper left", title="training", title_fontsize=6.4)
    c.set_title("(c) perturbation size", loc="left", fontsize=8.2)
    fig.savefig(os.path.join(OUT, "b_fig_gap.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_sweep(R):
    sw = R["R8_sweep"]["detectors"]
    if "xgb:natural" not in sw:
        return
    rows_by_fam = collections.defaultdict(list)
    for r in sw["xgb:natural"]["per_level"]:
        rows_by_fam[r["family"]].append(r)
    other = {k: {(r["family"], r["knob"], r["level"]): r for r in sw[k]["per_level"]} for k in sw}
    fams = [f for f in FAMS if rows_by_fam.get(f)]
    heights = [len(rows_by_fam[f]) for f in fams]
    fig, axs = plt.subplots(1, len(fams), figsize=(7.1, 0.11 * max(heights) + 1.6), gridspec_kw={"wspace": 1.15})
    cols = [("func", "functional"), ("xgb:natural", "XGB nat."), ("xgb:knob", "XGB knob"), ("iso:benign_only", "IF")]
    for ax, f in zip(np.atleast_1d(axs), fams):
        rr = sorted(rows_by_fam[f], key=lambda r: (r["knob"] != "base", r["knob"], r["level"] if r["level"] is not None else -1))
        labels = [("base" if r["knob"] == "base" else f"{r['knob'].replace('_pps', '').replace('_bytes', '').replace('_insert', '').replace('netem_', '')}={r['level']:g}") for r in rr]
        M = np.full((len(rr), len(cols)), np.nan)
        for i, r in enumerate(rr):
            M[i, 0] = r.get("func_frac", r.get("func"))
            for j, (k, _) in enumerate(cols[1:], start=1):
                o = other.get(k, {}).get((r["family"], r["knob"], r["level"]))
                if o is not None:
                    M[i, j] = o["flow_evasion"]
        ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]) and (j == 0 or M[i, j] > 0.005):
                    ax.text(j, i, "%.0f" % (100 * M[i, j]), ha="center", va="center", fontsize=4.6, color="white" if M[i, j] > 0.6 else "black")
        ax.set_yticks(range(len(rr))); ax.set_yticklabels(labels, fontsize=4.8)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], rotation=60, ha="right", fontsize=5.6)
        ax.set_title(NICE[f], fontsize=7.6)
        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    fig.suptitle("Targeted sweep: % of runs functional (first column) and % of attack flows called benign, per knob level", fontsize=7.6, x=0.01, ha="left")
    fig.savefig(os.path.join(OUT, "b_fig_sweep.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_mechanism(R):
    R3 = R["R3_feature_space"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.2, 1.1]})
    keys = [("attacker", "attacker-forward (src2dst)"), ("whole", "whole-flow (bidirectional)"), ("victim", "victim-side (dst2src)")]
    xs = []; labels = []
    pos = 0
    for regime in ("natural", "knob"):
        for f in FAMS:
            xs.append(pos); labels.append(NICE[f]); pos += 1
        pos += 0.8
    for gi, (g, lab) in enumerate(keys):
        vals = [100 * R3[f"xgb:{rg}"].get(f, {}).get("displacement_share", {}).get(g, 0) for rg in ("natural", "knob") for f in FAMS]
        prev = np.zeros(len(xs))
        for gg, _ in keys[:gi]:
            prev += np.array([100 * R3[f"xgb:{rg}"].get(f, {}).get("displacement_share", {}).get(gg, 0) for rg in ("natural", "knob") for f in FAMS])
        a.bar(xs, vals, 0.75, bottom=prev, color=C[g], label=lab)
    a.set_xticks(xs); a.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.8)
    a.text(2, 104, "natural training", ha="center", fontsize=6.8); a.text(7.8, 104, "knob-augmented", ha="center", fontsize=6.8)
    a.set_ylim(0, 112); a.set_ylabel("share of required displacement (%)")
    a.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2)
    a.set_title("(a) what the feature-space attack must move (XGBoost)", loc="left", fontsize=8.2)
    allf = []
    for regime in ("natural", "knob"):
        for n, _ in R.get(f"R4_full_top_features_{regime}", [])[:5]:
            if n not in allf:
                allf.append(n)
    y = np.arange(len(allf))[::-1]
    for regime, col, off in (("natural", "#9ecae1", 0.2), ("knob", "#3182bd", -0.2)):
        d = dict(R.get(f"R4_full_top_features_{regime}", []))
        b.barh(y + off, [d.get(n, 0) for n in allf], 0.4, color=col, label=RNAME[regime])
    b.set_yticks(y); b.set_yticklabels(allf, fontsize=6.6)
    for tl, n in zip(b.get_yticklabels(), allf):
        tl.set_color({"attacker": C["attacker"], "whole": "#B8650A", "victim": C["victim"]}[group_of(n)])
    b.set_xlabel("XGBoost gain importance (fold model)"); b.legend(frameon=False, fontsize=6.6, loc="lower right")
    b.set_title("(b) what the detector relies on", loc="left", fontsize=8.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_mechanism.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_ablation(R):
    R4 = R.get("R4_ablation", {}); R5 = R.get("R5_controllability", {}); R2 = R["R2_detectors"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    sets = [("full", "all features"), ("drop_top2", "drop top-2"), ("drop_victim_side", "drop victim-side"),
            ("no_size_features", "no size features"), ("log2_quantised", "log2-quantised")]
    cols = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#B279A2"]
    x = np.arange(len(FAMS)); w = 0.15
    for i, ((k, lab), col) in enumerate(zip(sets, cols)):
        src = R2["xgb:natural"] if k == "full" else R4.get(f"xgb:natural:{k}")
        if not src:
            continue
        v = [100 * (src["realised"].get(f, {}).get("r_rand") or 0) for f in FAMS]
        a.bar(x + (i - 2) * w, v, w, color=col, label=lab)
    a.set_xticks(x); a.set_xticklabels([NICE[f] for f in FAMS], rotation=30, ha="right")
    a.set_ylabel("functional flows called benign (%)"); a.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper right")
    a.set_title("(a) natural-trained XGBoost, feature ablations", loc="left", fontsize=8.2)
    fams = [f for f in FAMS if f in R5]
    for j, f in enumerate(fams):
        v = R5[f]["dst2src_bytes"]
        b.plot([j, j], [max(1, v["attack_p05"]), max(1, v["attack_p95"])], color=C["victim"], lw=4, solid_capstyle="butt")
        b.plot([j], [max(1, v["attack_median"])], "o", color="white", ms=3.5, mec=C["victim"])
    lo, hi = R.get("R5_benign_quantiles", {}).get("dst2src_bytes", (1, 1))
    b.axhspan(max(1, lo), max(1, hi), color=C["benign"], alpha=0.3, lw=0, label="benign 1st--99th pct.")
    b.set_yscale("log"); b.set_xticks(range(len(fams))); b.set_xticklabels([NICE[f] for f in fams], rotation=30, ha="right")
    b.set_ylabel("dst2src_bytes per flow (5th--95th pct.)"); b.legend(frameon=False, fontsize=6.4, loc="upper left")
    b.set_title("(b) victim-side bytes of functional flows", loc="left", fontsize=8.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_ablation.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_perturb(R):
    P = R["R7_perturbability"]["per_feature"]
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    rng = np.random.default_rng(0)
    for g, col in (("attacker", C["attacker"]), ("whole", C["whole"]), ("victim", C["victim"])):
        fs = [c for c in P if group_of(c) == g]
        ax.scatter([P[c]["PS"] + rng.uniform(-0.012, 0.012) for c in fs], [P[c]["executed_reach"] for c in fs], s=12, color=col, alpha=0.8, lw=0,
                   label={"attacker": "attacker-forward", "whole": "whole-flow", "victim": "victim-side"}[g])
    ax.axvline(0.87, color="#555555", ls=":", lw=0.8); ax.text(0.875, 0.98, "high", fontsize=6.4, color="#555555", va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("analytic perturb-ability score (PS)"); ax.set_ylabel("executed reachability (mean max $|\\rho_s|$)")
    ax.set_title(f"Spearman $\\rho$ = {R['R7_perturbability']['spearman_ps_vs_reach']:.2f}", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_perturb.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_gotham(R):
    R6 = R["R6_gotham"]; br = R6.get("benign_realism", {})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
    if br.get("ks_after"):
        aft = sorted(br["ks_after"].values())
        a.plot(range(len(aft)), aft, color=C["transfer"], lw=1.4, label=f"calibrated twin (median {np.median(aft):.2f})")
        a.fill_between(range(len(aft)), aft, 0, color=C["transfer"], alpha=0.12, lw=0)
        if br.get("ks_before"):
            bef = sorted(br["ks_before"].values())
            a.plot(range(len(bef)), bef, color=C["fs"], lw=1.4, label=f"uncalibrated twin (median {np.median(bef):.2f})")
        worst = sorted(br["ks_after"], key=br["ks_after"].get)[-4:][::-1]
        a.text(0.30, 0.74, "largest residual KS after calibration:\n" + "\n".join(f"{w.replace('bidirectional_', 'bi_').replace('_ms', '')}  {br['ks_after'][w]:.2f}" for w in worst),
               transform=a.transAxes, fontsize=5.9, va="top", ha="left", color="#333333")
        a.set_xlabel("flow feature (each curve sorted)"); a.set_ylabel("KS distance to real Gotham benign"); a.set_ylim(0, 1.05)
        a.legend(frameon=False, loc="upper left", fontsize=6.4)
        a.set_title(f"(a) benign realism; discriminator AUROC {br.get('discriminator_auc', float('nan')):.2f}", loc="left", fontsize=8)
    t = R6["twin_detector_on_gotham"]; g = R6["gotham_detector_on_twin"]
    labs = ["real\nattacks\nrecall", "real\nbenign\nFPR", "twin\nattacks\nrecall", "twin\nbenign\nFPR"]
    v1 = [t["at_twin_threshold"]["recall_real_attack"], t["at_twin_threshold"]["fpr_real_benign"]]
    v2 = [t["recalibrated_1pct_on_real_benign"]["recall_real_attack"], t["recalibrated_1pct_on_real_benign"]["fpr_real_benign_heldout"]]
    v3 = [np.mean(list(g["recall_twin_natural"].values())), g["fpr_twin_benign"]]
    xx = np.arange(2); xg = np.array([2.5, 3.5])
    b.bar(xx - 0.2, [100 * v for v in v1], 0.38, color="#9ecae1", label="twin detector, twin threshold")
    b.bar(xx + 0.2, [100 * v for v in v2], 0.38, color="#3182bd", label="twin detector, 1% real-FPR threshold")
    b.bar(xg, [100 * v for v in v3], 0.5, color=C["hsja"], label="Gotham detector on twin")
    for xv, v in list(zip(xx - 0.2, v1)) + list(zip(xx + 0.2, v2)) + list(zip(xg, v3)):
        b.text(xv, 100 * v + 2, f"{100 * v:.0f}", ha="center", fontsize=6.2)
    b.axvline(1.75, color="#BBBBBB", lw=0.6, ls=":")
    b.set_xticks(list(xx) + list(xg)); b.set_xticklabels(labs, fontsize=6.2)
    b.set_ylim(0, 125); b.set_ylabel("%"); b.legend(frameon=False, fontsize=6.0, loc="upper left")
    b.set_title(f"(b) transfer in both directions (twin$\\rightarrow$real AUROC {t['auc']:.2f})", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_fig_gotham.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
