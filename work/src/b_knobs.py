"""Paper B - attacker-controlled host-level knob catalogue and a quasi-random (Sobol) sampler.

Every knob is something an attacker who controls only its own bot/host can set: traffic-shaping on
its own NIC (tc netem), packet-crafting fields it originates (TTL, TCP window, source port, fragmentation,
payload padding), and how it structures its own sessions (rate, packets per connection, parallelism,
idle insertion, decoy cover traffic). None of them touches the victim, a router, or a captured trace.
"""
import hashlib, json
import numpy as np

# name: (low, high, kind)  kind in {log, lin, int, choicewidth}
KNOBS = {
    "rate_pps":      (1.0, 400.0, "log"),     # offered packets/sec (attacker send rate)
    "ipd_jitter":    (0.0, 1.0,   "lin"),     # 0..1 -> fraction of mean IPD used as uniform jitter
    "pad_bytes":     (0.0, 512.0, "int"),     # payload padding appended to each packet
    "pkts_per_conn": (1.0, 40.0,  "int"),     # packets before a new connection (TCP families)
    "parallel":      (1.0, 8.0,   "int"),     # concurrent streams / sources
    "srcport_mode":  (0.0, 3.0,   "int"),     # 0 fixed, 1 sequential, 2 random-high, 3 random-any
    "ttl":           (32.0, 128.0, "int"),    # IP TTL the attacker stamps
    "tcp_window":    (512.0, 65535.0, "log"), # advertised TCP window
    "frag_mtu":      (0.0, 3.0,   "int"),     # 0 none, 1 1500, 2 576, 3 256 (IP fragmentation)
    "netem_delay":   (0.0, 100.0, "lin"),     # ms egress delay on the attacker NIC
    "netem_loss":    (0.0, 5.0,   "lin"),     # % egress loss on the attacker NIC
    "idle_insert":   (0.0, 3.0,   "lin"),     # s idle gap inserted every burst
}
# v2: `cover` (probability of interleaving benign-looking cover packets) was sampled in v1 but no attack
# implementation ever read it, so it changed no traffic; it is no longer sampled and is recorded as 0.
NAMES = list(KNOBS)


def _sobol(n, d, seed):
    try:
        from scipy.stats import qmc
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n, d))


def sample(n, seed=0):
    """Return n knob configurations (list of dicts) spanning the catalogue."""
    U = _sobol(n, len(NAMES), seed)
    out = []
    for u in U:
        cfg = {}
        for j, name in enumerate(NAMES):
            lo, hi, kind = KNOBS[name]
            x = u[j]
            if kind == "log":
                v = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
            else:
                v = float(lo + x * (hi - lo))
            if kind == "int":
                v = int(round(v))
            cfg[name] = v
        cfg["cover"] = 0.0
        out.append(cfg)
    return out


def default():
    """The 'natural' configuration (fast, no evasion) used for clean training/eval."""
    return {"rate_pps": 200.0, "ipd_jitter": 0.0, "pad_bytes": 0, "pkts_per_conn": 1, "parallel": 1,
            "srcport_mode": 2, "ttl": 64, "tcp_window": 29200.0, "frag_mtu": 0, "netem_delay": 0.0,
            "netem_loss": 0.0, "idle_insert": 0.0, "cover": 0.0}


def cfg_id(family, cfg, rep):
    s = family + "|" + json.dumps(cfg, sort_keys=True) + f"|r{rep}"
    return hashlib.md5(s.encode()).hexdigest()[:16]


if __name__ == "__main__":
    cs = sample(4, 0)
    for c in cs:
        print(c)
