"""Paper B - Mininet IoT digital twin with real MQTT/CoAP/TCP services, sanitised attack families
parameterised by the knob catalogue, functionality oracles, and per-run packet capture.

Ethics: every attack is an inert, rate-capped re-implementation of a traffic pattern (scan, credential
guessing, SYN/UDP flood, CoAP reflection, HTTP beacon) run inside an isolated emulated network with no
real malware and no external connectivity. Payloads carry no exploit.

Run as root inside WSL (needs OVS + netns). One Twin instance keeps the network up; call run_attack()
many times with different knob configs for speed. Host processes use the host python3 (scapy/paho/aiocoap).
"""
import json, os, socket, subprocess, time
import numpy as np
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

setLogLevel("output")
HELP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b_hosts.py")   # staged beside this file

# IP plan
BROKER, VICTIM, SINK, ATTACKER = "10.0.0.10", "10.0.0.11", "10.0.0.12", "10.0.0.100"
IOT = [f"10.0.0.{20+i}" for i in range(4)]
OPEN_TCP = [80, 1883, 2323]                      # http, mqtt(broker via victim proxy), auth-svc
SCAN_PORTS = [22, 23, 80, 443, 1883, 2323, 8080, 8883]   # mix of open/closed


class Twin:
    def __init__(self, cap_dir):
        self.cap_dir = cap_dir
        os.makedirs(cap_dir, exist_ok=True)
        self.net = None

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        subprocess.run(["pkill", "-9", "-f", "b_hosts.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["mn", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        net = Mininet(switch=OVSSwitch, link=TCLink, controller=None)
        self.broker = net.addHost("broker", ip=BROKER + "/24")
        self.victim = net.addHost("victim", ip=VICTIM + "/24")
        self.sink = net.addHost("sink", ip=SINK + "/24")
        self.attacker = net.addHost("attacker", ip=ATTACKER + "/24")
        self.iot = [net.addHost(f"iot{i}", ip=ip + "/24") for i, ip in enumerate(IOT)]
        s1 = net.addSwitch("s1", failMode="standalone")
        for h in [self.broker, self.victim, self.sink, self.attacker] + self.iot:
            net.addLink(h, s1)
        net.start()
        self.net = net
        for h in [self.broker, self.victim, self.sink, self.attacker] + self.iot:
            i = h.defaultIntf().name
            h.cmd(f"ethtool -K {i} tso off gso off gro off tx off rx off 2>/dev/null")
        # shape the victim access link so a rate-capped flood can actually saturate it (10 Mbit)
        self.victim.cmd(f"tc qdisc add dev {self.victim.defaultIntf().name} root tbf rate 10mbit burst 32kbit latency 50ms 2>/dev/null")
        # make the victim realistically vulnerable to SYN exhaustion so the flood's effect is measurable
        self.victim.cmd("sysctl -w net.ipv4.tcp_syncookies=0 >/dev/null 2>&1")
        self.victim.cmd("sysctl -w net.ipv4.tcp_max_syn_backlog=8 >/dev/null 2>&1")
        self.victim.cmd("sysctl -w net.ipv4.tcp_synack_retries=5 >/dev/null 2>&1")
        self.victim.cmd("sysctl -w net.ipv4.somaxconn=8 >/dev/null 2>&1")
        self._start_services()
        return self

    def _start_services(self):
        # MQTT broker (amqtt) with auth; auth TCP service; CoAP server; HTTP sink; benign IoT clients
        self.broker.cmd(f"nohup python3 {HELP} mqtt_broker >/tmp/broker.log 2>&1 &")
        self.victim.cmd(f"nohup python3 {HELP} authsvc >/tmp/authsvc.log 2>&1 &")
        self.victim.cmd(f"nohup python3 {HELP} coapsrv >/tmp/coap.log 2>&1 &")
        self.victim.cmd(f"nohup python3 {HELP} httpsrv 80 >/tmp/http.log 2>&1 &")
        self.sink.cmd(f"nohup python3 {HELP} httpsrv 80 >/tmp/sink.log 2>&1 &")
        self.sink.cmd(f"nohup python3 {HELP} dns_server >/tmp/dns.log 2>&1 &")   # Gotham-calibrated benign
        self.sink.cmd(f"nohup python3 {HELP} ntp_server >/tmp/ntp.log 2>&1 &")
        time.sleep(4)

    def benign_capture(self, path, dev=0, dur=6.0):
        """Capture Gotham-calibrated benign traffic at one IoT device's interface (Gotham's per-device vantage)."""
        self.start_benign()
        self._cap_start(path, self.iot[dev % len(self.iot)])
        time.sleep(dur)
        self._cap_stop(self.iot[dev % len(self.iot)])
        self.stop_benign()

    def start_benign(self):
        for i, h in enumerate(self.iot):
            h.cmd(f"nohup python3 {HELP} benign {BROKER} {VICTIM} {i} >/tmp/benign{i}.log 2>&1 &")

    def stop_benign(self):
        for h in self.iot:
            h.cmd("pkill -f 'b_hosts.py benign' 2>/dev/null")

    def paired_goodput(self, dur):
        """No-attack goodput reference measured immediately before a flood run under IDENTICAL conditions: same
        probe host (iot0), benign traffic running, and the victim capture running (v1 used one average from another
        host, ~30% low; the capture also costs the victim CPU, so it must be on during the reference too)."""
        self.start_benign()
        self._cap_start("/tmp/goodput_ref.pcap", self.victim)
        self.iot[0].cmd(f"python3 {HELP} goodput {VICTIM} 80 {dur} >/tmp/pref.txt 2>&1")
        self._cap_stop(self.victim)
        self.stop_benign()
        try:
            return float(self.iot[0].cmd("cat /tmp/pref.txt").strip().splitlines()[-1])
        except Exception:
            return None

    def stop(self):
        if self.net:
            self.net.stop()
            self.net = None

    # ---------------------------------------------------------------- knobs on the attacker NIC
    def _apply_netem(self, cfg):
        i = self.attacker.defaultIntf().name
        self.attacker.cmd(f"tc qdisc del dev {i} root 2>/dev/null")
        d, l = cfg.get("netem_delay", 0), cfg.get("netem_loss", 0)
        if d > 0.5 or l > 0.05:
            j = max(0.0, d * cfg.get("ipd_jitter", 0))
            self.attacker.cmd(f"tc qdisc add dev {i} root netem delay {d:.1f}ms {j:.1f}ms loss {l:.2f}%")

    def _clear_netem(self):
        i = self.attacker.defaultIntf().name
        self.attacker.cmd(f"tc qdisc del dev {i} root 2>/dev/null")

    # ---------------------------------------------------------------- capture
    # libpcap >= 1.5 with TPACKET_V3 hands packets to tcpdump in ~1 s blocks unless immediate mode is set, so a
    # capture killed right after a short attack silently loses its tail (v1 lost most short brute-force runs).
    # v2: immediate mode, a drain pause before stopping, SIGINT, and wait until tcpdump has exited and flushed.
    CAP_DRAIN_S = 1.5

    def _cap_start(self, path, host):
        iface = host.defaultIntf().name
        host.cmd(f"rm -f '{path}'; nohup tcpdump -i {iface} --immediate-mode -B 8192 -w '{path}' -U -s 0 'ip' "
                 f">/tmp/tcpd.log 2>&1 & echo $! > /tmp/tcpd.pid")
        time.sleep(1.2)

    def _cap_stop(self, host):
        host.cmd(f"sleep {self.CAP_DRAIN_S}; P=$(cat /tmp/tcpd.pid); kill -INT $P 2>/dev/null; "
                 "for i in $(seq 50); do kill -0 $P 2>/dev/null || break; sleep 0.1; done")

    # ---------------------------------------------------------------- one attack run
    def run_attack(self, family, cfg, pcap, duration=6.0, with_benign=True):
        """Execute `family` under knob config `cfg`, capture at the targeted endpoint, return a record
        with the functionality metric and attacker cost. pcap is written to self.cap_dir/pcap."""
        tgt = self.sink if family == "httpbeacon" else self.victim
        path = os.path.join(self.cap_dir, pcap)
        self._apply_netem(cfg)
        if with_benign:
            self.start_benign()
        self._cap_start(path, tgt)
        t0 = time.time()
        func = self._launch(family, cfg, duration)
        wall = time.time() - t0
        self._cap_stop(tgt)
        if with_benign:
            self.stop_benign()
        self._clear_netem()
        rec = {"family": family, "cfg": cfg, "pcap": os.path.join("pcaps", pcap), "wall_s": round(wall, 2)}
        rec.update(func)
        return rec

    def _launch(self, family, cfg, duration):
        a = self.attacker
        args = json.dumps(cfg)
        if family == "netscan":
            out = a.cmd(f"python3 {HELP} atk_netscan {VICTIM} '{args}' {duration}")
        elif family == "svcbrute":
            out = a.cmd(f"python3 {HELP} atk_svcbrute {VICTIM} '{args}' {duration}")
        elif family == "synflood":
            # A SYN flood only exhausts the victim's (syncookie-off, backlog-8) listen queue if the half-open
            # entries are not torn down. From its own address the attacker's kernel answers each unsolicited
            # SYN-ACK with a RST, which frees the slot, so the flood barely bit. The attacker drops its own
            # outbound RSTs (a host-level action it controls) for the duration of the flood, so the half-open
            # entries persist and legitimate connections are refused. This is scoped to the flood: the other
            # TCP families need normal connection teardown.
            a.cmd("iptables -A OUTPUT -p tcp --tcp-flags RST RST -j DROP")
            self.iot[0].cmd(f"nohup python3 {HELP} goodput {VICTIM} 80 {duration} >/tmp/good.txt 2>&1 &")
            out = a.cmd(f"python3 {HELP} atk_synflood {VICTIM} '{args}' {duration}")
            time.sleep(0.5)
            a.cmd("iptables -D OUTPUT -p tcp --tcp-flags RST RST -j DROP 2>/dev/null")
            g = self.iot[0].cmd("cat /tmp/good.txt 2>/dev/null").strip()
            try:
                out = json.dumps({"goodput_req": float(g.splitlines()[-1])})
            except Exception:
                out = '{"goodput_req": 0.0}'
        elif family == "coapamp":
            out = a.cmd(f"python3 {HELP} atk_coapamp {VICTIM} {SINK} '{args}' {duration}")
        elif family == "httpbeacon":
            out = a.cmd(f"python3 {HELP} atk_httpbeacon {SINK} '{args}' {duration}")
        else:
            raise ValueError(family)
        try:
            return json.loads(out.strip().splitlines()[-1])
        except Exception:
            return {"func_raw": out.strip()[-200:], "func_ok": 0}


if __name__ == "__main__":
    import b_knobs
    t = Twin("/tmp/btwin_test").start()
    print("services up; pinging core")
    print(t.net.ping([t.attacker, t.victim], timeout="2"))
    for fam in ("netscan", "svcbrute", "synflood", "coapamp", "httpbeacon"):
        r = t.run_attack(fam, b_knobs.default(), f"test_{fam}.pcap", duration=5)
        sz = os.path.getsize(os.path.join(t.cap_dir, f"test_{fam}.pcap"))
        print(fam, "->", {k: v for k, v in r.items() if k != "cfg"}, "pcap", sz, "B")
    t.stop()
