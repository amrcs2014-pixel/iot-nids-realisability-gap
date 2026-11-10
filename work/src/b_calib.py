"""Generate Gotham-calibrated benign-only captures (per-device vantage) and extract their flows.
Runs in WSL (Mininet + nfstream). Writes data/btwin/benign_cal_flows.parquet (label 0)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from nfstream import NFStreamer
from b_twin import Twin

ROOT = "/mnt/e/network research/p 15,16/work"
DATA = os.path.join(ROOT, "data", "btwin"); CAP = os.path.join(DATA, "benign_cal")
OUT = os.path.join(DATA, "benign_cal_flows.parquet")
N = int(os.environ.get("N_CAL", 40)); DUR = float(os.environ.get("DUR", 8.0))


def main():
    os.makedirs(CAP, exist_ok=True)
    if os.path.exists(OUT) and all(os.path.exists(os.path.join(CAP, f"cal_{k:03d}.pcap")) for k in range(N)):
        print("calibrated benign already complete", flush=True); return
    todo = [k for k in range(N) if not os.path.exists(os.path.join(CAP, f"cal_{k:03d}.pcap"))]   # resumable
    if todo:
        t = Twin(CAP).start()
        print(f"twin up (calibrated benign), {len(todo)} captures to do", flush=True)
        try:
            for k in todo:
                p = os.path.join(CAP, f"cal_{k:03d}.pcap")
                t.benign_capture(p + ".part", dev=k, dur=DUR)
                os.replace(p + ".part", p)
                if k % 10 == 0:
                    print(f"  {k}/{N}", flush=True)
        finally:
            t.stop()
    frames = []
    for k in range(N):
        p = os.path.join(CAP, f"cal_{k:03d}.pcap")
        if not os.path.exists(p) or os.path.getsize(p) < 40:
            continue
        try:
            df = NFStreamer(source=p, statistical_analysis=True, idle_timeout=15, active_timeout=120).to_pandas()
        except Exception as e:
            print("nfstream fail", k, repr(e)[:60], flush=True); continue
        if df is not None and len(df):
            df["label"] = 0; df["family"] = "benign"; df["run_id"] = f"cal_{k:03d}"; df["kind"] = "benign_cal"
            frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(OUT, index=False)
        import collections
        print(f"calibrated benign flows: {len(out)}; proto {dict(out['protocol'].value_counts())}; "
              f"dst_port top {dict(out['dst_port'].value_counts().head(8))}", flush=True)
        print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
