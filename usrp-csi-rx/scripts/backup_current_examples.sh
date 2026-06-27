#!/usr/bin/env bash
set -e

BACKUP_PATH="$HOME/gr_ieee80211_examples_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "[BACKUP] Creating backup at: $BACKUP_PATH"

cd "$HOME/gr-ieee802-11"
tar -czf "$BACKUP_PATH" examples

echo "[BACKUP] Done."
ls -lh "$BACKUP_PATH"
