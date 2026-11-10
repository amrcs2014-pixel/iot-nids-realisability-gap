# iot-nids-realisability-gap

Analysis code for measuring how much feature-space adversarial evasion of flow-based IoT intrusion detectors can be
realised as traffic by a host-only attacker, in a reproducible Mininet/Containernet + Open vSwitch digital twin.

## Contents (`work/src/`)
- `b_twin.py`, `b_hosts.py` — the IoT twin (MQTT, CoAP, HTTP services; benign device profiles; attacker host)
- `b_knobs.py`, `b_gen.py`, `b_calib.py` — attacker knob catalogue, traffic generation, benign calibration
- `b_flows.py`, `b_gotham_extract.py`, `remotezip.py` — nfstream flow extraction (twin and reference traces)
- `b_detect.py`, `b_attack.py`, `b_gotham.py`, `b_failure.py` — detectors, attacks, external-validity analysis
- `b_report.py` — renders figures and tables from the result files

## Data
Reference traffic: GothamDataset2025 (Zenodo 14502760, CC BY 4.0), not redistributed here. All attacks are inert,
rate-capped re-implementations run inside an isolated emulated network.
