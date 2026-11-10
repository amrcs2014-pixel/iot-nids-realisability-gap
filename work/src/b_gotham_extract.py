"""Paper B external-validity data: pull a representative sample of real GothamDataset2025 pcaps (via HTTP
range, no full download), extract flows with the SAME nfstream extractor as the twin, label by folder, and
write gotham_flows.parquet. Runs in WSL (nfstream + libpcap). Resumable by member name.
"""
import io, os, sys, time
import numpy as np
import pandas as pd
from nfstream import NFStreamer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remotezip import HttpFile
import zipfile

URL = "https://zenodo.org/api/records/14502760/files/GothamDataset2025.zip/content"
OUT = "/mnt/e/network research/p 15,16/work/data/btwin/gotham_flows.parquet"
TMP = "/dev/shm/gotham_tmp.pcap"
# small, representative members: benign + the attack families that match the twin (scan, amplification, C2, dos)
MEMBERS = [
    ("raw/benign/iotsim-building-monitor-1_0-0_to_OpenvSwitch-28_1-0.pcap", "benign"),
    ("raw/benign/iotsim-domotic-monitor-1_0-0_to_OpenvSwitch-23_1-0.pcap", "benign"),
    ("raw/benign/iotsim-air-quality-1_0-0_to_OpenvSwitch-25_1-0.pcap", "benign"),
    ("raw/benign/iotsim-city-power-1_0-0_to_OpenvSwitch-26_1-0.pcap", "benign"),
    ("raw/benign/iotsim-hydraulic-system-1_0-0_to_OpenvSwitch-15_1-0.pcap", "benign"),
    ("raw/benign/iotsim-cooler-motor-1_0-0_to_OpenvSwitch-18_1-0.pcap", "benign"),
    ("raw/malicious/network-scanning/iotsim-building-monitor-1_0-0_to_OpenvSwitch-28_1-0.pcap", "netscan"),
    ("raw/malicious/network-scanning/iotsim-domotic-monitor-1_0-0_to_OpenvSwitch-23_1-0.pcap", "netscan"),
    ("raw/malicious/network-scanning/iotsim-air-quality-1_0-0_to_OpenvSwitch-25_1-0.pcap", "netscan"),
    ("raw/malicious/network-scanning/iotsim-combined-cycle-1_0-0_to_OpenvSwitch-13_1-0.pcap", "netscan"),
    ("raw/malicious/network-scanning/iotsim-city-power-1_0-0_to_OpenvSwitch-26_1-0.pcap", "netscan"),
    ("raw/malicious/coap-amplificator/iotsim-combined-cycle-1_0-0_to_OpenvSwitch-13_1-0.pcap", "coapamp"),
    ("raw/malicious/merlin/iotsim-city-power-1_0-0_to_OpenvSwitch-26_1-0.pcap", "merlin_c2"),
    ("raw/malicious/merlin/iotsim-domotic-monitor-1_0-0_to_OpenvSwitch-23_1-0.pcap", "merlin_c2"),
    ("raw/malicious/mirai-infection/iotsim-air-quality-1_0-0_to_OpenvSwitch-25_1-0.pcap", "mirai_infect"),
    ("raw/malicious/mirai-infection/iotsim-city-power-1_0-0_to_OpenvSwitch-26_1-0.pcap", "mirai_infect"),
    ("raw/malicious/mirai-dos/iotsim-air-quality-1_0-0_to_OpenvSwitch-25_1-0.pcap", "synflood"),
]


def main():
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 20))
    done = set()
    if os.path.exists(OUT):
        try:
            done = set(pd.read_parquet(OUT, columns=["member"])["member"].unique())
        except Exception:
            done = set()
    frames = []
    for mem, fam in MEMBERS:
        if mem in done:
            continue
        t0 = time.time()
        try:
            with z.open(mem) as src, open(TMP, "wb") as dst:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    dst.write(b)
        except Exception as e:
            print("fetch fail", mem, repr(e)[:80], flush=True); continue
        try:
            df = NFStreamer(source=TMP, statistical_analysis=True, idle_timeout=15, active_timeout=120).to_pandas()
        except Exception as e:
            print("nfstream fail", mem, repr(e)[:80], flush=True); continue
        if df is None or len(df) == 0:
            continue
        df["label"] = 0 if fam == "benign" else 1
        df["family"] = fam
        df["member"] = mem
        frames.append(df)
        print(f"{mem.split('/')[-1][:40]} {fam}: {len(df)} flows in {time.time()-t0:.0f}s", flush=True)
    if not frames:
        print("nothing new"); return
    new = pd.concat(frames, ignore_index=True)
    if os.path.exists(OUT):
        try:
            new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
        except Exception:
            pass
    new.to_parquet(OUT, index=False)
    print(f"gotham flows: {len(new)}; by family: {new['family'].value_counts().to_dict()}", flush=True)
    if os.path.exists(TMP):
        os.remove(TMP)


if __name__ == "__main__":
    main()
