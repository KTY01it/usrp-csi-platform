# USRP CSI Platform

A two-machine USRP B210 platform for WiFi-like CSI sensing experiments.

## Structure

- `usrp-csi-rx/`: Receiver-side GNU Radio and CSI processing code.
- `usrp-csi-tx/`: Transmitter-side GNU Radio code.
- `shared/`: Shared dataset schema, protocol notes, and experiment definitions.

## Machine roles

- Rx machine works only inside `usrp-csi-rx/`.
- Tx machine works only inside `usrp-csi-tx/`.
- Shared experiment definitions are stored in `shared/`.

## Current stage

- Rx baseline is initialized.
- Tx folder is a placeholder and should be populated from the Tx machine.
- Data files are ignored by Git.

## Rx quick start

```bash
cd usrp-csi-rx
./scripts/run_rx.sh
