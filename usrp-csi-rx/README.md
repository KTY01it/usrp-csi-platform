# USRP B210 2x2 MIMO-CSI Receiver and Controller

RX-side implementation and experiment controller for the USRP B210 MIMO-CSI sensing platform.

This directory contains the receiver, CSI extraction pipeline, 2x2 physical TDM MIMO tensor construction, sensing feature extraction, multi-view acquisition, geometry registration, dataset quality control, and dataset finalization tools.

For the complete system architecture, scientific validity policy, data format, research references, and overall platform documentation, see the repository-level `README.md`.

---

## 1. Role in the platform

The RX computer is the primary controller of the production sensing system.

It performs:

- dual-RX USRP B210 reception
- IEEE 802.11 OFDM synchronization and equalization
- CSI extraction
- physical TX0/TX1 TDM pairing
- 2x2 MIMO CSI tensor construction
- RF sample-domain timing
- per-view quality control
- angular RF evidence extraction
- CIR/PDP and delay-domain feature extraction
- temporal motion feature extraction
- unified sensing feature export
- manual multi-view pose registration
- dataset indexing
- nominal antenna geometry transformation
- spatial fusion input construction
- final dataset validation

The TX implementation is located in:

    ../usrp-csi-tx/

---

## 2. Production branch

The canonical integrated production version is:

    main

The historical machine-specific branch:

    rx-machine

is retained for RX development history and traceability.

For real data collection, both TX and RX computers should use the same production `main` revision whenever possible.

---

## 3. Hardware

Validated RX hardware:

    USRP B210
    RX0
    RX1

Current nominal RX antenna geometry:

    RX0 -> RX1 spacing: 0.040 m

The configured value represents nominal physical antenna-center spacing.

RF phase centers have not been independently calibrated.

At approximately 5.89 GHz, 40 mm is greater than lambda/2, so spatial aliasing is possible. The production system therefore does not claim validated absolute AoA solely from the nominal array geometry.

---

## 4. Check the platform

From the RX computer:

    cd ~/Documents/usrp-csi-platform/usrp-csi-rx

Run:

    ./csi-sense doctor
    ./csi-sense status

The `doctor` command checks the active platform configuration, required software, USRP availability, and calibration references.

The `status` command displays the active sensing platform configuration.

---

## 5. Collect a real multi-view dataset

Start a production scan with:

    ./csi-sense scan <dataset_name> \
      --duration 30 \
      --snr-min 10 \
      --tx-interval-ms 100

Example:

    ./csi-sense scan room01 \
      --duration 30 \
      --snr-min 10 \
      --tx-interval-ms 100

The RX controller normally starts and stops the TX process remotely.

For each view, register the actual TX and RX platform pose in the common `room_local` coordinate system.

A pose contains:

    x
    y
    z
    yaw
    pitch
    roll

Use controlled stop-and-go acquisition: register the pose, place the hardware at that pose, keep the setup stationary during the capture, and then move to the next view.

---

## 6. View acceptance

A production view is accepted only when acquisition quality control passes.

The current primary requirements include:

    minimum complete TDM cycles: 80
    physical TX mapping valid
    finite CSI on all required links

With the validated 100 ms TX packet cadence, a 30 s capture normally provides sufficient margin.

A short capture may contain valid CSI but still be rejected if it does not contain enough complete TX0/TX1 TDM cycles.

Rejected views should not be included in the primary production dataset.

---

## 7. Finalize the dataset

After all views have been collected, run:

    ./csi-sense finalize <session_path>

The finalization pipeline performs:

    dataset QC
        ->
    dataset index
        ->
    global nominal antenna geometry
        ->
    spatial fusion input

A production-ready dataset must end with:

    dataset status: PASS
    FINALIZATION: PASS
    DATASET READY: True

---

## 8. Data location

All canonical sensing datasets are stored on the RX computer under:

    ~/Documents/usrp-csi-platform/usrp-csi-rx/sessions/

Typical structure:

    sessions/
    └── <dataset_name>/
        └── <dataset_name>_YYYYMMDD_HHMMSS/
            ├── manifest.json
            ├── dataset_qc.json
            ├── dataset_index.npz
            ├── global_geometry.npz
            ├── spatial_fusion_input.npz
            ├── config_snapshot/
            └── views/
                ├── view_000/
                │   ├── view_manifest.json
                │   ├── csi/
                │   ├── features/
                │   ├── quality/
                │   └── logs/
                ├── view_001/
                └── ...

Dataset files are ignored by Git.

---

## 9. Physical 2x2 MIMO CSI tensor

The primary per-view CSI artifact is:

    csi/H_raw_tdm_physical_2x2.npz

Logical tensor shape:

    H[N, 52, 2, 2]

Physical mapping:

    H[n,:,0,0] = RX0 <- TX0
    H[n,:,1,0] = RX1 <- TX0
    H[n,:,0,1] = RX0 <- TX1
    H[n,:,1,1] = RX1 <- TX1

The TX identity comes from deterministic packet-level TDM:

    even packet -> TX0
    odd packet  -> TX1

---

## 10. Per-view sensing features

Accepted views are automatically processed into:

    features/angular_features.npz
    features/delay_features.npz
    features/temporal_motion_features.npz
    features/sensing_features.npz

Interpretation:

- `angular_features.npz`: subcarrier-resolved inter-RX angular RF evidence and confidence
- `delay_features.npz`: CIR/PDP and delay-domain evidence
- `temporal_motion_features.npz`: temporal CSI / relative phase-motion evidence
- `sensing_features.npz`: unified sensing feature package

These files contain measurements and RF evidence for downstream estimators. They are not automatically absolute physical ground truth.

---

## 11. Session-level artifacts

Important finalized artifacts include:

    dataset_index.npz
    global_geometry.npz
    spatial_fusion_input.npz
    dataset_qc.json
    manifest.json

`dataset_index.npz` references accepted multi-view measurements.

`global_geometry.npz` contains registered nominal physical antenna-center coordinates.

`spatial_fusion_input.npz` associates CSI measurements, sensing evidence, timing, view identity, and nominal TX/RX geometry for downstream AI, localization, or reconstruction models.

---

## 12. Main production components

Core RX production components include:

    csi-sense
    scripts/run_rx_tdm_view.sh

    tools/build_tdm_mimo_tensor.py
    tools/build_angular_features.py
    tools/mimo_cir_pdp.py
    tools/build_temporal_motion_features.py
    tools/build_unified_sensing_features.py
    tools/run_view_feature_pipeline.py

    tools/session_manager.py
    tools/validate_multiview_dataset.py
    tools/build_multiview_dataset_index.py
    tools/build_multiview_global_geometry.py
    tools/build_spatial_fusion_input.py

The repository also contains legacy and research-diagnostic tools. Their presence does not mean they are part of the canonical production acquisition path.

---

## 13. Scientific scope

The RX pipeline provides valid measured or derived quantities including:

    complex CSI
    amplitude
    RF sample-domain timing
    relative phase
    angular RF evidence
    CIR/PDP and delay evidence
    temporal motion evidence
    nominal registered antenna geometry

The current production system does not claim independently validated absolute:

    AoA
    AoD
    ToF
    range
    Doppler
    velocity
    target position
    radar-grade backprojection

Those quantities require additional calibration, estimation, learning, or independent physical validation.

The production pipeline deliberately keeps measured RF evidence separate from downstream physical estimators.

---

## 14. Related documentation

For the complete platform documentation, including:

- system architecture
- TX/RX interaction
- data collection protocol
- dataset schema
- downstream data usage
- scientific validity policy
- research basis and referenced papers
- reproducibility requirements

see:

    ../README.md
