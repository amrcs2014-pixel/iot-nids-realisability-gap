"""Paper B v2 pipeline completeness check: exit 0 only when every planned library / calibration / sweep run has a
record and every record has been extracted to flows. Prints a one-line status either way."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("N_SOBOL", "120"); os.environ.setdefault("REP_DEFAULT", "12"); os.environ.setdefault("N_BENIGN", "24")
import b_gen, b_sweep

D = b_gen.DATA


def ids(p):
    s = set()
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try:
                s.add(json.loads(l)["id"])
            except Exception:
                pass
    return s


def flow_runs(p):
    try:
        return set(pd.read_parquet(p, columns=["run_id"])["run_id"].unique())
    except Exception:
        return set()


lib_plan = {j[0] for j in b_gen.plan()}; lib_done = ids(b_gen.RUNS) & lib_plan
sw_plan = {j[0] for j in b_sweep.plan()}; sw_done = ids(b_sweep.RUNS) & sw_plan
lib_fl = flow_runs(os.path.join(D, "flows.parquet")); sw_fl = flow_runs(os.path.join(D, "sweep_flows.parquet"))
cal_ok = os.path.exists(os.path.join(D, "benign_cal_flows.parquet"))
st = {"library": f"{len(lib_done)}/{len(lib_plan)}", "lib_extracted": f"{len(lib_fl & lib_done)}/{len(lib_done)}",
      "calibrated_benign": cal_ok, "sweep": f"{len(sw_done)}/{len(sw_plan)}", "sweep_extracted": f"{len(sw_fl & sw_done)}/{len(sw_done)}"}
complete = (lib_done == lib_plan and sw_done == sw_plan and cal_ok
            and lib_done <= lib_fl | {i for i in lib_done if i.startswith("benign_")} and sw_done <= sw_fl)
print("V2 STATUS", st, "COMPLETE" if complete else "incomplete", flush=True)
sys.exit(0 if complete else 1)
