"""Paper B - knob-library generation driver (resumable, runs the twin and caches pcaps + run records).

Produces, into data/btwin/:
  pcaps/<id>.pcap           one capture per run
  b_runs.jsonl              one record per run: family, cfg, pcap, functionality metric, attacker cost
  benign/<k>.pcap           benign-only captures (negative class / clean training)

Config budget per family: the default (natural) config repeated REP_DEFAULT times (noise floor + clean
positives) + a Sobol knob-library of N_SOBOL configs. Benign-only captures for the negative class.
Resumable: run ids already in b_runs.jsonl are skipped, so the 10-min scheduled task self-heals.
"""
import json, os, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_knobs
from b_twin import Twin

ROOT = "/mnt/e/network research/p 15,16/work"
DATA = os.path.join(ROOT, "data", "btwin")
PCAPS = os.path.join(DATA, "pcaps"); BEN = os.path.join(DATA, "benign")
RUNS = os.path.join(DATA, "b_runs.jsonl")
LOG = os.path.join(ROOT, "logs", "twingen.log")
FAMILIES = ["netscan", "svcbrute", "synflood", "coapamp", "httpbeacon"]
N_SOBOL = int(os.environ.get("N_SOBOL", 120))
REP_DEFAULT = int(os.environ.get("REP_DEFAULT", 12))
N_BENIGN = int(os.environ.get("N_BENIGN", 24))
DUR = float(os.environ.get("DUR", 6.0))
# v2: per-family wall-clock budget. Scan and brute-force END when done, so a 6-s cap marked every slow
# ("low-and-slow") configuration non-functional by the clock (v1: brute 1/120 Sobol functional); they now get
# 30 s. Flood, reflection and beaconing run for exactly their budget.
BUDGET = {"netscan": float(os.environ.get("SLOW_BUDGET", 30.0)), "svcbrute": float(os.environ.get("SLOW_BUDGET", 30.0)),
          "synflood": DUR, "coapamp": DUR, "httpbeacon": DUR}


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


def append(rec):
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def plan():
    """Ordered list of (id, kind, family, cfg, rep). kind in {benign, attack}."""
    jobs = []
    for k in range(N_BENIGN):
        jobs.append((f"benign_{k:03d}", "benign", None, None, k))
    for fam in FAMILIES:
        for r in range(REP_DEFAULT):
            jobs.append((b_knobs.cfg_id(fam, b_knobs.default(), r), "attack", fam, b_knobs.default(), r))
        for j, cfg in enumerate(b_knobs.sample(N_SOBOL, seed=zlib.crc32(fam.encode()) & 0xffff)):
            jobs.append((b_knobs.cfg_id(fam, cfg, 0), "attack", fam, cfg, 0))
    return jobs


def main():
    os.makedirs(PCAPS, exist_ok=True); os.makedirs(BEN, exist_ok=True)
    jobs = [j for j in plan() if j[0] not in done_ids()]
    if not jobs:
        log("ALL DONE: nothing to generate"); return
    log(f"{len(jobs)} runs to generate (of {len(plan())} planned)")
    t = Twin(PCAPS).start()
    log("twin up")
    baseline = None
    try:
        for n, (rid, kind, fam, cfg, rep) in enumerate(jobs):
            try:
                if kind == "benign":
                    path = os.path.join(BEN, rid + ".pcap")
                    t.start_benign(); t._cap_start(path, t.victim)
                    time.sleep(DUR)
                    # also capture broker-side benign by a second pass is unnecessary; victim sees MQTT? no.
                    t._cap_stop(t.victim); t.stop_benign()
                    rec = {"id": rid, "kind": "benign", "pcap": os.path.join("benign", rid + ".pcap")}
                else:
                    pref = t.paired_goodput(BUDGET[fam]) if fam == "synflood" else None
                    rec = t.run_attack(fam, cfg, rid + ".pcap", duration=BUDGET[fam], with_benign=True)
                    rec["id"] = rid; rec["kind"] = "attack"; rec["budget_s"] = BUDGET[fam]
                    if fam == "synflood":
                        if baseline is None:
                            baseline = _baseline_goodput(t)
                        rec["goodput_baseline_v1"] = baseline
                        rec["goodput_paired"] = pref
                        ref = pref if pref else baseline
                        rec["goodput_drop"] = round(1 - rec.get("goodput_req", 0) / max(1.0, ref), 3)
                        rec["func_ok"] = int(rec["goodput_drop"] >= 0.2)
                append(rec)
                if n % 20 == 0:
                    log(f"{n}/{len(jobs)} done (last {rid} {fam or 'benign'})")
            except Exception as e:
                log(f"ERR {rid}: {repr(e)[:160]}")
                time.sleep(1)
    finally:
        t.stop()
    log(f"batch complete; total records now {len(done_ids())}")


def _baseline_goodput(t):
    """Average several no-attack goodput probes for a stable reference."""
    hp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b_hosts.py")
    t.start_benign(); vals = []
    for _ in range(3):
        t.iot[1].cmd(f"python3 {hp} goodput {t.victim.IP()} 80 {DUR} >/tmp/base.txt 2>&1")
        try:
            vals.append(float(t.iot[1].cmd("cat /tmp/base.txt").strip().splitlines()[-1]))
        except Exception:
            pass
    t.stop_benign()
    return (sum(vals) / len(vals)) if vals else 1.0


if __name__ == "__main__":
    main()
