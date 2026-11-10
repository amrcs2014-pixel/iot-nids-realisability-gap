"""Paper B - extract bidirectional flows from every cached pcap (nfstream), label them, attach the run's
family / functionality / knob settings, and write one flow table. Runs in WSL (nfstream + libpcap).

Labelling (each pcap is a single run):
  benign pcap  -> every flow is benign.
  attack pcap  -> a flow is the run's attack family if it involves the attacker (10.0.0.100), or, for
                  coapamp, the reflected victim<->sink CoAP traffic (udp/5683); all other flows (the
                  benign IoT background captured at the same endpoint) are benign.
Identifiers (IPs/MACs/ports) are kept only for labelling and dropped from the ML feature set by b_detect.
"""
import json, os, sys
import numpy as np
import pandas as pd
from nfstream import NFStreamer

ROOT = "/mnt/e/network research/p 15,16/work"
DATA = os.path.join(ROOT, "data", "btwin")
RUNS = os.environ.get("B_RUNS", os.path.join(DATA, "b_runs.jsonl"))        # the targeted sweep uses
OUT = os.environ.get("B_OUT", os.path.join(DATA, "flows.parquet"))         # sweep/sweep_runs.jsonl -> sweep_flows.parquet
ATTACKER, VICTIM, SINK, BROKER = "10.0.0.100", "10.0.0.11", "10.0.0.12", "10.0.0.10"
KNOBS = ["rate_pps", "ipd_jitter", "pad_bytes", "pkts_per_conn", "parallel", "srcport_mode",
         "ttl", "tcp_window", "frag_mtu", "netem_delay", "netem_loss", "idle_insert", "cover"]


def flow_is_attack(r, family):
    s, d, dp, proto = r["src_ip"], r["dst_ip"], r["dst_port"], r["protocol"]
    if ATTACKER in (s, d):
        return True
    if family == "coapamp" and proto == 17 and (dp == 5683 or r["src_port"] == 5683) and {s, d} <= {VICTIM, SINK, ATTACKER}:
        return True
    return False


def extract(rec):
    path = os.path.join(DATA, rec["pcap"])
    if not os.path.exists(path) or os.path.getsize(path) < 40:
        return None
    try:
        df = NFStreamer(source=path, statistical_analysis=True, idle_timeout=15, active_timeout=120).to_pandas()
    except Exception as e:
        print("nfstream fail", rec["id"], repr(e)[:80], flush=True); return None
    if df is None or len(df) == 0:
        return None
    fam = rec.get("family")
    if rec["kind"] == "benign":
        df["label"] = 0; df["family"] = "benign"
    else:
        mask = df.apply(lambda r: flow_is_attack(r, fam), axis=1)
        df["label"] = mask.astype(int)
        df["family"] = np.where(mask, fam, "benign")
    df["run_id"] = rec["id"]; df["kind"] = rec["kind"]
    df["func_ok"] = rec.get("func_ok", 1)
    if "sweep_knob" in rec:
        df["sweep_knob"] = rec["sweep_knob"]; df["sweep_level"] = float(rec["sweep_level"]) if rec["sweep_level"] is not None else np.nan
        df["goodput_drop"] = float(rec.get("goodput_drop", np.nan) if rec.get("goodput_drop") is not None else np.nan)
    cfg = rec.get("cfg") or {}
    for k in KNOBS:
        df["k_" + k] = cfg.get(k, np.nan)
    return df


def main():
    recs = []
    for l in open(RUNS, encoding="utf-8"):
        try:
            recs.append(json.loads(l))
        except Exception:
            pass        # skip a partial last line while the generator is still writing
    done = set()
    if os.path.exists(OUT):
        try:
            prev = pd.read_parquet(OUT, columns=["run_id"]); done = set(prev["run_id"].unique())
        except Exception:
            done = set()
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(todo)} new runs to extract ({len(done)} already in table)", flush=True)
    frames = []
    for i, r in enumerate(todo):
        d = extract(r)
        if d is not None:
            frames.append(d)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    if not frames:
        print("nothing new"); return
    new = pd.concat(frames, ignore_index=True)
    if os.path.exists(OUT):
        try:
            new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
        except Exception:
            pass
    new.to_parquet(OUT, index=False)
    print(f"flows: {len(new)} rows, {new['label'].mean():.3f} attack; by family:\n{new['family'].value_counts()}", flush=True)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
