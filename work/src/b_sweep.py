"""Paper B - targeted, functionality-preserving knob sweep (runs in WSL, resumable).

One-factor-at-a-time from a FUNCTIONAL base configuration per family: every knob that the family's attack
actually implements is varied over several levels while all other knobs stay at the base. Scan and brute-force
get a generous time budget so that slow ("low-and-slow") configurations can complete instead of being marked
non-functional by the clock. Each configuration is executed REPS times. Output: data/btwin/sweep/pcaps/*.pcap and
data/btwin/sweep/sweep_runs.jsonl (same record format as b_runs.jsonl, plus sweep_knob / sweep_level).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_knobs
from b_twin import Twin

ROOT = "/mnt/e/network research/p 15,16/work"
DATA = os.path.join(ROOT, "data", "btwin")
SW = os.path.join(DATA, "sweep"); PCAPS = os.path.join(SW, "pcaps")
RUNS = os.path.join(SW, "sweep_runs.jsonl")
LOG = os.path.join(ROOT, "logs", "sweep.log")
REPS = int(os.environ.get("REPS", 2)); BASE_REPS = int(os.environ.get("BASE_REPS", 4))

# knobs each attack implementation actually reads (netem delay/loss are applied by the harness to every family)
FAMILY_KNOBS = {
    "netscan": ["rate_pps", "ipd_jitter", "pad_bytes", "srcport_mode", "ttl", "tcp_window", "idle_insert", "netem_delay", "netem_loss"],
    "svcbrute": ["rate_pps", "ipd_jitter", "pad_bytes", "pkts_per_conn", "idle_insert", "netem_delay", "netem_loss"],
    "synflood": ["rate_pps", "ipd_jitter", "pad_bytes", "parallel", "srcport_mode", "ttl", "tcp_window", "frag_mtu",
                 "idle_insert", "netem_delay", "netem_loss"],
    "coapamp": ["rate_pps", "ipd_jitter", "srcport_mode", "ttl", "idle_insert", "netem_delay", "netem_loss"],
    "httpbeacon": ["rate_pps", "ipd_jitter", "pad_bytes", "netem_delay", "netem_loss"],
}
LEVELS = {"rate_pps": [2, 10, 50, 200, 400], "ipd_jitter": [0.25, 0.5, 1.0], "pad_bytes": [0, 16, 64, 128, 256, 512],
          "pkts_per_conn": [2, 5, 10, 20, 40], "parallel": [1, 2, 4, 8], "srcport_mode": [0, 1, 3], "ttl": [32, 128],
          "tcp_window": [512, 4096, 65535], "frag_mtu": [1, 2, 3], "netem_delay": [10, 50, 100],
          "netem_loss": [1, 3, 5], "idle_insert": [0.5, 1.5, 3.0]}
BUDGET = {"netscan": 120.0, "svcbrute": 120.0, "synflood": 6.0, "coapamp": 6.0, "httpbeacon": 6.0}


def base_cfg(fam):
    c = dict(b_knobs.default()); c["cover"] = 0.0
    # the flood exhausts the victim's 8-entry, syncookie-off SYN backlog (the attacker drops its own outbound RSTs,
    # so half-open entries persist). The natural single-sender flood with random high source ports already fills
    # the backlog (drop ~0.8), so it is a valid functional base like the other families.
    return c


def plan():
    jobs = []
    for fam, knobs in FAMILY_KNOBS.items():
        b = base_cfg(fam)
        for r in range(BASE_REPS):
            jobs.append((f"sw_{fam}_base_r{r}", fam, b, "base", None))
        for k in knobs:
            for lv in LEVELS[k]:
                if abs(float(b.get(k, 0)) - float(lv)) < 1e-9:
                    continue
                c = dict(b); c[k] = lv
                for r in range(REPS):
                    jobs.append((f"sw_{fam}_{k}_{lv}_r{r}", fam, c, k, lv))
    return jobs


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + m
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def done_ids():
    s = set()
    if os.path.exists(RUNS):
        for line in open(RUNS, encoding="utf-8"):
            try:
                s.add(json.loads(line)["id"])
            except Exception:
                pass
    return s


def main():
    os.makedirs(PCAPS, exist_ok=True)
    jobs = [j for j in plan() if j[0] not in done_ids()]
    log(f"sweep: {len(jobs)} runs to do (of {len(plan())} planned)")
    if not jobs:
        log("SWEEP ALL DONE"); return
    import b_gen
    t = Twin(PCAPS).start(); log("twin up")
    baseline = None
    try:
        for n, (rid, fam, cfg, knob, lv) in enumerate(jobs):
            try:
                pref = t.paired_goodput(BUDGET[fam]) if fam == "synflood" else None
                rec = t.run_attack(fam, cfg, rid + ".pcap", duration=BUDGET[fam], with_benign=True)
                rec.update({"id": rid, "kind": "attack", "pcap": os.path.join("sweep", "pcaps", rid + ".pcap"),
                            "sweep_knob": knob, "sweep_level": lv, "budget_s": BUDGET[fam]})
                if fam == "synflood":
                    if baseline is None:
                        baseline = b_gen._baseline_goodput(t)
                    rec["goodput_baseline_v1"] = baseline
                    rec["goodput_paired"] = pref
                    ref = pref if pref else baseline
                    rec["goodput_drop"] = round(1 - rec.get("goodput_req", 0) / max(1.0, ref), 3)
                    rec["func_ok"] = int(rec["goodput_drop"] >= 0.2)
                with open(RUNS, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n % 10 == 0:
                    log(f"{n}/{len(jobs)} {rid} func={rec.get('func_ok')} wall={rec.get('wall_s')}")
            except Exception as e:
                log(f"ERR {rid}: {repr(e)[:160]}"); time.sleep(1)
    finally:
        t.stop()
    log(f"batch complete; sweep records now {len(done_ids())}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plan":
        p = plan(); import collections
        print(len(p), "runs;", dict(collections.Counter(j[1] for j in p)))
    else:
        main()
