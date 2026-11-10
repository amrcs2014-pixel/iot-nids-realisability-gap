"""Paper B - capture-completeness check (runs in WSL). Executes short attacks with the v1 capture stop (immediate
kill, block-buffered libpcap) and with the v2 capture stop (immediate mode + drain + SIGINT), and counts the
attacker packets that reach the pcap against what the attack reports it sent. Writes logs/captest.json."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_knobs
from b_twin import Twin, ATTACKER

ROOT = "/mnt/e/network research/p 15,16/work"
OUT = os.path.join(ROOT, "logs", "captest.json")
CAP = "/tmp/captest"


def v1_start(self, path, host):
    iface = host.defaultIntf().name
    host.cmd(f"rm -f '{path}'; nohup tcpdump -i {iface} -w '{path}' -U -s 0 'ip' >/dev/null 2>&1 & echo $! > /tmp/tcpd.pid")
    time.sleep(1.2)


def v1_stop(self, host):
    host.cmd("kill $(cat /tmp/tcpd.pid) 2>/dev/null; sleep 0.3")


def count(path):
    r = subprocess.run(["tcpdump", "-nr", path, "host", ATTACKER], capture_output=True, text=True)
    return len([l for l in r.stdout.splitlines() if l.strip()])


def main():
    t = Twin(CAP).start()
    v2_start, v2_stop = Twin._cap_start, Twin._cap_stop
    res = []
    try:
        for fam, dur in (("svcbrute", 6.0), ("netscan", 6.0), ("httpbeacon", 6.0)):
            for mode in ("v1", "v2"):
                Twin._cap_start, Twin._cap_stop = (v1_start, v1_stop) if mode == "v1" else (v2_start, v2_stop)
                for r in range(3):
                    rid = f"{fam}_{mode}_{r}.pcap"
                    rec = t.run_attack(fam, dict(b_knobs.default()), rid, duration=dur, with_benign=True)
                    n = count(os.path.join(CAP, rid))
                    res.append({"family": fam, "mode": mode, "rep": r, "attacker_pkts": n, "wall_s": rec.get("wall_s"),
                                "attempts": rec.get("attempts"), "sent": rec.get("sent"), "func_ok": rec.get("func_ok")})
                    print(res[-1], flush=True)
    finally:
        Twin._cap_start, Twin._cap_stop = v2_start, v2_stop
        t.stop()
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
