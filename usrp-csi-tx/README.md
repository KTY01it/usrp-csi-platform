# USRP CSI Tx

Tx-side project for USRP B210 WiFi-like CSI sensing experiments.

## Current stage

- SISO WiFi Tx baseline.
- gr-ieee802-11 based packet transmission.
- Future 2Tx time-division pilot support.
- Future Tx metadata and pilot scheduling.

## Run

```bash
./scripts/run_tx.sh
```

## Notes

This is the Tx-side folder only. The Rx machine should work inside `usrp-csi-rx/`.
