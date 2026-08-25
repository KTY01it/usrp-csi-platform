# USRP B210 2x2 TDM MIMO-CSI Transmitter

TX-side implementation of the USRP B210 MIMO-CSI sensing platform.

This directory contains the transmitter software used together with the RX/controller implementation in `usrp-csi-rx/`.

For the complete system architecture, data format, scientific validity policy, references, and multi-view collection workflow, see the repository-level `README.md`.

---

## 1. Role in the platform

The TX machine provides IEEE 802.11-style OFDM packet transmission using one USRP B210 with two physical transmit RF chains.

Production sensing uses packet-level time-division multiplexing:

    even packet -> physical TX0
    odd packet  -> physical TX1

The RX side uses this deterministic packet mapping to reconstruct the physical 2x2 MIMO CSI tensor:

    H[N, 52, 2, 2]

The TX machine does not perform dataset management, sensing feature extraction, or multi-view geometry processing. Those functions are controlled by the RX machine.

---

## 2. Production branch

The canonical integrated production version is:

    main

The historical machine-specific branch:

    tx-machine

is retained for TX development history and traceability.

For real data collection, both TX and RX machines should use the same production `main` revision whenever possible.

---

## 3. Hardware

Validated hardware:

    USRP B210
    2 physical TX RF chains
    TX0
    TX1

Current nominal TX antenna geometry:

    TX0 -> TX1 spacing: 0.042 m

The configured value represents nominal physical antenna-center spacing.

RF phase centers have not been independently calibrated.

---

## 4. Production TDM transmitter

From the TX machine:

    cd ~/Documents/usrp-csi-platform/usrp-csi-tx

Start the production 2TX TDM transmitter manually with:

    TDM_ACTIVE_TX=both \
    TDM_INTERVAL_MS=100 \
    ./scripts/run_tx_tdm_2x2.sh

Production mapping:

    packet 0 -> TX0
    packet 1 -> TX1
    packet 2 -> TX0
    packet 3 -> TX1
    ...

The normal real-data configuration is:

    TDM_ACTIVE_TX=both
    TDM_INTERVAL_MS=100

---

## 5. TX selection modes

TX0 only:

    TDM_ACTIVE_TX=0 \
    TDM_INTERVAL_MS=100 \
    ./scripts/run_tx_tdm_2x2.sh

TX1 only:

    TDM_ACTIVE_TX=1 \
    TDM_INTERVAL_MS=100 \
    ./scripts/run_tx_tdm_2x2.sh

Both transmit chains:

    TDM_ACTIVE_TX=both \
    TDM_INTERVAL_MS=100 \
    ./scripts/run_tx_tdm_2x2.sh

Production datasets normally use:

    both

---

## 6. Packet cadence

The packet interval is controlled by:

    TDM_INTERVAL_MS

The validated production collection setting is:

    100 ms

A 20 ms high-cadence mode has also been tested for temporal sensing.

High packet cadence alone does not establish physically validated Doppler or velocity measurements.

---

## 7. Remote control from RX

During normal multi-view collection, the experiment is normally started from the RX/controller machine:

    ./csi-sense scan <dataset_name> \
      --duration 30 \
      --snr-min 10 \
      --tx-interval-ms 100

The RX orchestration layer starts and stops the TX remotely.

Manual TX startup is primarily for:

- hardware testing
- TX-chain isolation
- debugging
- standalone RF validation

It is not normally necessary during an automated production scan.

---

## 8. Main TX components

Production TX components:

    apps/wifi_tx_tdm_2x2.py
    apps/tdm_packet_router.py
    scripts/run_tx_tdm_2x2.sh

### `wifi_tx_tdm_2x2.py`

Builds and runs the 2TX OFDM transmitter flowgraph.

### `tdm_packet_router.py`

Routes successive packets according to the physical TDM rule:

    even -> TX0
    odd  -> TX1

### `run_tx_tdm_2x2.sh`

Production launcher that prepares the runtime environment and starts the TDM transmitter.

---

## 9. Legacy and experimental files

Other TX applications and scripts may remain in this directory for historical experiments, hardware validation, or controlled research tests.

Examples include:

    wifi_tx_2tx_legacy.py
    wifi_tx_2tx_aod_legacy.py
    wifi_tx_trackCSI_legacy.py
    wifi_tx_ch1*.py
    tx_tone_ch0.py
    tx_tone_ch1.py

These files should not be assumed to be part of the current production acquisition path.

The canonical production TX path is:

    apps/wifi_tx_tdm_2x2.py
    apps/tdm_packet_router.py
    scripts/run_tx_tdm_2x2.sh

---

## 10. Data storage

The TX machine does not generate the canonical sensing dataset.

TX runtime logs may be stored under:

    usrp-csi-tx/logs/

The CSI dataset is collected and organized on the RX machine under:

    usrp-csi-rx/sessions/

See the repository-level `README.md` for the complete dataset schema and downstream usage.

---

## 11. Scientific scope

The TX implementation provides controlled OFDM illumination and deterministic physical TX identity.

It does not directly provide ground-truth:

    AoA
    AoD
    ToF
    range
    Doppler
    velocity
    target position

Those quantities require downstream processing, calibration, estimation, or independent validation.

The TX side should therefore be interpreted as the controlled signal-generation component of the complete MIMO-CSI sensing platform.
