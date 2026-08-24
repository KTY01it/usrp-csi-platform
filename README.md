# USRP B210 2x2 MIMO-CSI Sensing Platform

A two-machine software-defined radio platform for controlled WiFi-like CSI sensing experiments using two Ettus/NI USRP B210 devices. The platform is designed as a **controlled ISAC-like MIMO-OFDM CSI measurement system**: one machine is dedicated to transmission, one machine is dedicated to reception, and the RX side orchestrates multi-view data collection, quality control, feature extraction, geometry registration, dataset indexing, and finalization.

> Current scientific scope: calibrated/relative CSI sensing and radar-inspired RF evidence extraction. The platform does **not** currently claim physically validated absolute AoA, AoD, ToF/range, Doppler, velocity, or radar-grade backprojection.

## 1. Quick start

### 1.1 Production branch and machine roles

`main` is the canonical integrated production branch and contains both the RX/controller and TX implementations.

The repository also retains the machine-specific branches for development history and traceability:

- `rx-machine` -> RX/controller development branch
- `tx-machine` -> TX development branch

For production deployment, use `main` on both computers and run the appropriate subdirectory for each machine.

Recommended local paths:

```bash
~/Documents/usrp-csi-platform/usrp-csi-rx
~/Documents/usrp-csi-platform/usrp-csi-tx
```

The intended hardware is:

- 2 x USRP B210
- TX B210: 2 physical TX RF chains
- RX B210: 2 physical RX RF chains
- physical TDM mapping: even packet -> TX0, odd packet -> TX1
- 52 active OFDM subcarriers

The validated development configuration uses a center frequency around 5.89 GHz and a 5 MHz sample rate. Always check the active YAML configuration before collection.

### 1.2 TX machine

On the TX computer:

```bash
cd ~/Documents/usrp-csi-platform/usrp-csi-tx
```

Normal production scans are controlled remotely from the RX machine. For a manual TX test, the 2x2 TDM transmitter can be started with:

```bash
TDM_ACTIVE_TX=both \
TDM_INTERVAL_MS=100 \
./scripts/run_tx_tdm_2x2.sh
```

TDM semantics:

```text
even packet -> physical TX0
odd packet  -> physical TX1
```

`TDM_ACTIVE_TX=0` or `TDM_ACTIVE_TX=1` can be used for controlled channel-isolation tests. Production data collection normally uses `both`.

### 1.3 RX machine: check platform

On the RX/controller computer:

```bash
cd ~/Documents/usrp-csi-platform/usrp-csi-rx

./csi-sense doctor
./csi-sense status
```

`doctor` checks configuration files, required software, USRP discovery, and the active calibration references.

### 1.4 Collect a real multi-view dataset

Run:

```bash
./csi-sense scan <dataset_name> \
  --duration 30 \
  --snr-min 10 \
  --tx-interval-ms 100
```

Example:

```bash
./csi-sense scan room01 \
  --duration 30 \
  --snr-min 10 \
  --tx-interval-ms 100
```

For each view the tool asks for the manually registered TX and RX platform pose in the `room_local` coordinate system:

```text
TX: x, y, z, yaw, pitch, roll
RX: x, y, z, yaw, pitch, roll
```

After registering a pose, physically place the platforms at that pose and capture the view. Repeat for as many views as required. Use `q` when the scan is complete.

A view is accepted only when acquisition QC passes. The production criterion currently requires at least 80 complete TDM cycles, valid physical TX mapping, and finite CSI links. With the validated 100 ms TX cadence, 30 s/view normally provides sufficient margin; short 10 s smoke captures may be rejected simply because they contain too few complete TDM cycles.

### 1.5 Finalize the dataset

After collection:

```bash
./csi-sense finalize <session_path>
```

A usable production dataset must end with:

```text
dataset status: PASS
FINALIZATION: PASS
DATASET READY: True
```

Do not use rejected or non-finalized sessions as training data unless they are being analyzed specifically for debugging.

---

## 2. Where data are stored

All scan sessions are stored on the RX machine under:

```text
~/Documents/usrp-csi-platform/usrp-csi-rx/sessions/
```

A scan named `room01` creates a timestamped session similar to:

```text
sessions/
└── room01/
    └── room01_YYYYMMDD_HHMMSS/
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
```

### 2.1 Per-view CSI files

Important files inside `views/view_xxx/csi/` include:

```text
H_raw_tdm_physical_2x2.npz
csi_*.bin
csi_*.jsonl
```

The main physical TDM MIMO tensor is:

```text
H_raw_tdm_physical_2x2.npz
```

with the logical tensor shape:

```text
[N, 52, 2, 2]
```

where the current mapping is:

```python
H[n,:,0,0] = RX0 <- TX0
H[n,:,1,0] = RX1 <- TX0
H[n,:,0,1] = RX0 <- TX1
H[n,:,1,1] = RX1 <- TX1
```

### 2.2 Per-view feature files

Accepted views are post-processed automatically. The manifest records the primary artifacts:

```text
features/angular_features.npz
features/delay_features.npz
features/temporal_motion_features.npz
features/sensing_features.npz
```

Interpretation:

- `angular_features.npz`: inter-RX wideband angular RF evidence, confidence and validity information
- `delay_features.npz`: per-link CIR/PDP and delay-domain evidence
- `temporal_motion_features.npz`: time-varying CSI / phase-motion evidence
- `sensing_features.npz`: unified per-view feature package for downstream processing

These are estimator inputs, not absolute physical ground truth.

### 2.3 Session-level files

`dataset_index.npz` provides the multi-view dataset index and references the per-view measurements without forcing all views to have the same cycle count.

`global_geometry.npz` contains registered **nominal physical antenna-center coordinates** derived from the manually entered platform poses and configured local antenna geometry.

`spatial_fusion_input.npz` associates measurements, view identity, timing, and nominal geometry in one package for downstream spatial models.

`manifest.json` and each `view_manifest.json` store lifecycle metadata, pose registration, QC state, artifact paths, and scientific validity flags.

`config_snapshot/` preserves the configuration used for that session, so later code/config changes do not silently change the meaning of previously collected data.

### 2.4 Typical data use

For signal-processing research, use the per-view raw/CSI artifacts.

For AI estimation, localization, or reconstruction, a recommended starting point is:

```text
spatial_fusion_input.npz
+ per-view sensing_features.npz
+ dataset_index.npz
+ global_geometry.npz
```

A downstream model can combine:

- complex CSI and amplitude
- subcarrier-resolved relative phase
- angular confidence/evidence
- delay-domain descriptors
- temporal motion evidence
- TX/RX nominal geometry
- view pose and quality masks

The current platform intentionally keeps measurement/evidence separate from learned or model-based physical estimates.

---

## 3. System architecture

The production path is intentionally layered:

```text
TX waveform generation
        |
        v
2-TX packet-level TDM
        |
        v
USRP B210 RF transmission
        |
        v
wireless channel / scene
        |
        v
USRP B210 RX0 + RX1
        |
        v
IEEE 802.11 OFDM synchronization/equalization
        |
        v
per-packet CSI + metadata
        |
        v
physical TDM pairing -> H[N,52,2,2]
        |
        +--> angular RF evidence
        +--> delay / CIR / PDP evidence
        +--> temporal motion evidence
        |
        v
per-view unified features + QC
        |
        v
manual multi-view pose registration
        |
        v
dataset QC / index / nominal global geometry
        |
        v
spatial_fusion_input.npz
        |
        v
future AI estimators / localization / reconstruction
```

### 3.1 TX design

The TX side generates IEEE 802.11-style OFDM frames and routes alternating packets to the two B210 TX chains. Packet-level TDM was selected because it gives explicit physical TX identity while remaining implementable on two B210 RF channels without assuming phase-coherent simultaneous MIMO precoding.

The TX packet interval is configurable through `TDM_INTERVAL_MS`. The validated production collection setting is typically 100 ms. A 20 ms high-cadence mode was also tested for temporal sensing, but physical Doppler separation was not validated; therefore it is not treated as a Doppler ground-truth generator.

### 3.2 RX CSI extraction

The receiver is based on the GNU Radio `gr-ieee802-11` OFDM processing chain. CSI is taken from the equalizer/LTF channel estimate rather than from the retired custom all-ones estimator. The platform keeps 52 occupied OFDM carriers and associates packet CSI with decoded packet sequence information and RF sample-domain timing.

The sample-domain metadata field `rf_sample_index` is propagated through synchronization/equalization so relative RF time can be reconstructed as:

```text
t_rf_rel = (rf_sample_index - rf_sample_index_0) / Fs
```

### 3.3 Physical 2x2 TDM tensor

CSI packets from the alternating physical TX chains are paired into a tensor with explicit RX/TX dimensions. This prevents the later sensing pipeline from confusing packet parity, RF chain identity, and MIMO matrix indexing.

The tensor builder validates:

- complete TX0/TX1 cycles
- physical TX mapping
- finite complex CSI
- timing consistency

### 3.4 Angular evidence

For each TX, the primary inter-RX complex spatial evidence is formed from the two RX chains, conceptually:

```python
angular_complex = H_spatial[:, :, RX1, TX] * conj(H_spatial[:, :, RX0, TX])
```

The platform preserves subcarrier-resolved relative phase, coherence, confidence, and validity. TX0 is currently the primary angular evidence path; TX1 is auxiliary/quality-gated based on controlled validation.

The RX antenna spacing used in the current geometry configuration is 40 mm. At approximately 5.89 GHz this is greater than lambda/2, so spatial aliasing is possible. Controlled sweeps did not justify an absolute arcsin AoA claim. Consequently absolute AoA remains explicitly invalid in production metadata.

### 3.5 Delay evidence

The tool forms per-link CIR/PDP evidence from the OFDM CSI. At 5 MHz bandwidth, the native inverse-DFT delay-bin scale is approximately 200 ns; zero-padding only interpolates this response and does not create additional physical bandwidth or radar range resolution.

For that reason the platform stores delay-domain evidence but does not convert the delay-bin index directly to absolute range.

### 3.6 Temporal evidence

Temporal sensing uses repeated CSI measurements and relative RF sample-domain timing. Controlled motion changed the temporal CSI evidence, but the high-cadence experiment did not establish reliable spectral separation required for a physical Doppler claim. The production schema therefore distinguishes `temporal motion evidence` from `physical Doppler`.

### 3.7 Geometry and multi-view design

Each view records manual TX/RX pose in a common `room_local` coordinate system. Local nominal antenna coordinates are transformed into room coordinates using the registered platform translation and orientation.

Current nominal antenna separations are:

```text
RX0 -> RX1: 0.040 m
TX0 -> TX1: 0.042 m
```

These values represent measured/declared physical antenna-center geometry. RF phase centers have not been independently calibrated, so metadata keeps RF phase-center geometry and absolute spatial backprojection disabled.

This design supports controlled stop-and-go multi-view acquisition. It does not claim automatic trajectory tracking; if the platforms are moved continuously, an external pose source or another registration method would be needed for precise reconstruction.

---

## 4. Scientific validity policy

The repository deliberately distinguishes four classes of quantities:

1. **Measured**: complex CSI, packet metadata, timing, amplitude.
2. **Calibrated/relative evidence**: relative phase, angular evidence, differential delay evidence.
3. **Geometry hypotheses**: nominal antenna coordinates, candidate path geometry, broadside/array-axis descriptors.
4. **Estimated physical quantities**: AoA, AoD, ToF/range, Doppler, velocity, target position, occupancy/reconstruction.

The first three can be stored as model inputs when their validity flags pass. The fourth class must not be relabeled as measured ground truth unless an independent calibration/validation procedure supports it.

Current production status:

| Quantity | Status |
|---|---|
| complex 2x2 CSI | measured / valid |
| amplitude | measured / derived |
| relative phase | measured / calibrated evidence |
| angular RF evidence | validated evidence |
| absolute AoA | not validated |
| absolute AoD | not validated |
| CIR/PDP / delay evidence | available |
| physical ToF/range | not validated |
| temporal motion evidence | validated evidence |
| physical Doppler / velocity | not validated |
| nominal antenna-center geometry | available |
| RF phase-center geometry | not validated |
| physical range backprojection | disabled |

---

## 5. Design basis and related research

This repository is not a reimplementation of one paper. Its architecture combines an SDR IEEE 802.11 receiver foundation with CSI sensing ideas from several research lines. The relationship to each reference is listed explicitly below.

### 5.1 GNU Radio IEEE 802.11 PHY foundation

**B. Bloessl, M. Segata, C. Sommer, and F. Dressler, “An IEEE 802.11a/g/p OFDM Receiver for GNU Radio,” ACM SRIF/SIGCOMM 2013.**  
DOI: https://doi.org/10.1145/2491246.2491248

This work is the direct architectural basis for using an open GNU Radio IEEE 802.11 OFDM receiver with access to PHY-layer synchronization, equalization, and decoded frames. This repository extends that software path with CSI export, RF sample-domain timing, dual-RX capture, TDM TX pairing, sensing features, and dataset orchestration.

### 5.2 Super-resolution array / angular processing

**R. O. Schmidt, “Multiple Emitter Location and Signal Parameter Estimation,” IEEE Transactions on Antennas and Propagation, 1986.**  
DOI: https://doi.org/10.1109/TAP.1986.1143830

This is the classical MUSIC foundation relevant to the repository's experimental MUSIC/AoA tools. Production data collection does not assume that MUSIC automatically yields absolute AoA; array geometry, phase-center calibration, aliasing, and repeatability remain separate validation requirements.

### 5.3 WiFi CSI AoA/ToF processing

**M. Kotaru, K. Joshi, D. Bharadia, and S. Katti, “SpotFi: Decimeter Level Localization Using WiFi,” ACM SIGCOMM 2015.**  
DOI: https://doi.org/10.1145/2785956.2787487

SpotFi motivates exploiting CSI phase structure across antennas and subcarriers and using super-resolution processing for angular/delay inference. This repository uses the same general physical dimensions—antenna-domain and subcarrier-domain information—but does not claim SpotFi-equivalent calibration or accuracy.

### 5.4 ToF and bandwidth limitations

**D. Vasisht, S. Kumar, and D. Katabi, “Decimeter-Level Localization with a Single WiFi Access Point (Chronos),” USENIX NSDI 2016.**  
https://www.usenix.org/conference/nsdi16/technical-sessions/presentation/vasisht

Chronos is important to the design policy because accurate WiFi ToF requires substantially more frequency diversity than a naive single narrowband IFFT interpretation. The present 5 MHz acquisition therefore stores CIR/PDP/delay evidence but does not label an interpolated delay bin as physical sub-nanosecond ToF or radar range.

### 5.5 Joint multidimensional CSI sensing

**K. Qian, C. Wu, Y. Zhang, G. Zhang, Z. Yang, and Y. Liu, “Widar2.0: Passive Human Tracking with a Single Wi-Fi Link,” ACM MobiSys 2018.**  
DOI: https://doi.org/10.1145/3210240.3210314

Widar2.0 motivates treating AoA, ToF, Doppler/frequency shift, attenuation, and CSI cleaning as distinct but related sensing dimensions. This repository follows that separation concept by exporting angular, delay, and temporal evidence independently and fusing them only downstream. Unlike Widar2.0, current production metadata does not declare those physical parameters validated simply because a feature can be computed.

### 5.6 Measurement-platform architecture

**Z. Jiang et al., “Eliminating the Barriers: Demystifying Wi-Fi Baseband Design and Introducing the PicoScenes Wi-Fi Sensing Platform,” IEEE Internet of Things Journal.**  
DOI: https://doi.org/10.1109/JIOT.2021.3104666

PicoScenes motivates a measurement-platform mindset: preserve low-level CSI/baseband information, expose stable interfaces, separate hardware/measurement from downstream applications, and retain rich metadata. This project uses a different GNU Radio/USRP implementation but follows the same general principle of making the sensing dataset reusable rather than embedding one application directly in the capture path.

### 5.7 Object reconstruction research direction

**T.-D. Nguyen, N.-S. Duong, J.-Y. Pan, and V.-L. Nguyen, “DualSense: WiFi Sensing and Segmentation of Signal Scatterers For Object Reconstruction,” IEEE ICC 2026.**

DualSense motivates the downstream objective of transforming WiFi sensing evidence into object-level spatial/reconstruction representations. The acquisition platform in this repository is designed to provide stronger, explicitly registered real SDR evidence for that class of downstream learning/reconstruction research. The current USRP pipeline should not be described as a reproduction of DualSense; it is an experimental measurement platform intended to support subsequent estimator and reconstruction models.

---

## 6. Repository structure

```text
usrp-csi-platform/
├── usrp-csi-rx/
│   ├── csi-sense
│   ├── apps/
│   ├── configs/
│   ├── scripts/
│   ├── tools/
│   └── sessions/          # local collected data; ignored by Git
├── usrp-csi-tx/
│   ├── apps/
│   ├── configs/
│   ├── scripts/
│   └── tools/
└── shared/
```

Major RX production components include:

```text
csi-sense
scripts/run_rx_tdm_view.sh
tools/build_tdm_mimo_tensor.py
tools/run_view_feature_pipeline.py
tools/session_manager.py
tools/validate_multiview_dataset.py
tools/build_multiview_dataset_index.py
tools/build_multiview_global_geometry.py
tools/build_spatial_fusion_input.py
```

Major TX production components include:

```text
apps/wifi_tx_tdm_2x2.py
apps/tdm_packet_router.py
scripts/run_tx_tdm_2x2.sh
```

Legacy and research-diagnostic tools remain in the repository for controlled experiments, but they are not automatically treated as production physical estimators.

---

## 7. Recommended collection discipline

Before a real dataset campaign:

- freeze RX/TX software commits
- record antenna spacing and orientation
- define the room-local coordinate origin and axes
- define a repeatable stop-and-go trajectory
- avoid moving unregistered equipment during a view
- keep RF/sample-rate/gain/TDM settings constant unless the experiment explicitly studies them
- finalize every completed session
- retain only sessions with `DATASET READY: True` for the primary dataset

For reconstruction, collect multiple geometrically diverse views. Four instantaneous 2x2 bistatic links alone do not uniquely determine a detailed 3D scene; multi-view spatial diversity and known/registered poses are part of the intended design.

## 8. Reproducibility and Git branches

`main` is the canonical integrated production version of the platform. It contains both the RX/controller and TX implementations.

Repository layout:

    usrp-csi-platform/
    ├── usrp-csi-rx/    -> RX/controller implementation
    ├── usrp-csi-tx/    -> TX implementation
    ├── shared/          -> shared definitions/resources
    └── README.md

The machine-specific branches are retained for development traceability:

    rx-machine -> RX/controller development branch
    tx-machine -> TX development branch

For a production experiment, both computers should use the same integrated `main` revision. Record the exact Git commit SHA used during data collection together with the generated session manifest and configuration snapshot.

Dataset binaries and scan sessions are not committed to Git. Code, configuration templates, validation rules, and documentation are version controlled. Experimental datasets should remain local or be published separately together with their manifests, configuration snapshots, and software revision information.
