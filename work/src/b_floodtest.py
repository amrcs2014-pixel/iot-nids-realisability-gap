"""Paper B - flood-oracle check (runs in WSL): paired goodput reference vs goodput during (a) no attack, (b) the
natural flood, (c) a heavy flood, each twice. Writes logs/floodtest.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_knobs
from b_twin import Twin

OUT = "/mnt/e/network research/p 15,16/work/logs/floodtest.json"


def main():
    t = Twin("/tmp/floodtest").start()
    res = []
    nat = dict(b_knobs.default())
    heavy = dict(nat); heavy.update({"rate_pps": 400.0, "parallel": 8, "pad_bytes": 512})
    mid = dict(nat); mid.update({"rate_pps": 400.0, "parallel": 4, "pad_bytes": 256})
    try:
        for name, cfg in (("none", None), ("natural", nat), ("base4x400x256", mid), ("heavy8x400x512", heavy)):
            for r in range(2):
                ref = t.paired_goodput(6.0)
                if cfg is None:
                    att = t.paired_goodput(6.0)
                else:
                    rec = t.run_attack("synflood", cfg, f"{name}_{r}.pcap", duration=6.0, with_benign=True)
                    att = rec.get("goodput_req")
                res.append({"cond": name, "rep": r, "ref": ref, "during": att,
                            "drop": None if not ref else round(1 - (att or 0) / ref, 3)})
                print(res[-1], flush=True)
    finally:
        t.stop()
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
