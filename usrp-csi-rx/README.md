# USRP CSI Rx

Rx-only project for USRP B210 WiFi-like CSI sensing experiments.

## Current stage

This repository contains the receiver-side GNU Radio scripts and CSI processing tools.

Current baseline:

- SISO WiFi Rx
- gr-ieee802-11 based packet reception
- future raw IQ logging
- future CSI extraction
- future 1Tx-2Rx / 2Tx-2Rx MIMO CSI support

## Directory structure

```text
apps/      Receiver flowgraphs and runtime scripts
configs/   Rx machine configuration
tools/     CSI, AoA, ToF, calibration utilities
data/      Local captured data, ignored by Git
logs/      Local logs and metadata, ignored by Git
archive/   Local archived scripts
scripts/   Helper shell scripts
```

## Run

```bash
./scripts/run_rx.sh
```

## Notes

This is the Rx-side repository only. The Tx machine should use a separate Tx-only repository with matched frequency, sample rate, gain, and frame configuration.
