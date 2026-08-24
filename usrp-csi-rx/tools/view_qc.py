#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_view(root, view_dir):
    root = Path(root).resolve()
    view_dir = Path(view_dir).resolve()

    acquisition_cfg = load_yaml(
        root / "configs/platform/acquisition.yaml"
    )

    minimum_cfg = acquisition_cfg["capture"]["minimum"]

    minimum_cycles = int(
        minimum_cfg["complete_tdm_cycles"]
    )

    tensor_file = (
        view_dir
        / "csi"
        / "H_raw_tdm_physical_2x2.npz"
    )

    result = {
        "accepted": False,
        "reason": None,
        "complete_tdm_cycles": 0,
        "minimum_complete_tdm_cycles": minimum_cycles,
        "tensor_shape": None,
        "physical_tx_mapping_valid": False,
        "all_links_finite": False,
    }

    if not tensor_file.exists():
        result["reason"] = "missing_tdm_tensor"
        return result

    try:
        z = np.load(
            tensor_file,
            allow_pickle=False
        )
    except Exception:
        result["reason"] = "invalid_tdm_tensor"
        return result

    if "H_raw" not in z.files:
        result["reason"] = "missing_H_raw"
        return result

    H = z["H_raw"]

    result["tensor_shape"] = list(H.shape)

    if H.ndim != 4:
        result["reason"] = "invalid_tensor_rank"
        return result

    if H.shape[1:] != (52, 2, 2):
        result["reason"] = "invalid_tensor_shape"
        return result

    cycles = int(H.shape[0])

    result["complete_tdm_cycles"] = cycles

    if "physical_tx_mapping_valid" in z.files:
        mapping_valid = bool(
            z["physical_tx_mapping_valid"]
        )
    else:
        mapping_valid = False

    result[
        "physical_tx_mapping_valid"
    ] = mapping_valid

    finite = bool(
        np.all(np.isfinite(H.real))
        and np.all(np.isfinite(H.imag))
    )

    result[
        "all_links_finite"
    ] = finite

    if not mapping_valid:
        result["reason"] = "invalid_physical_tx_mapping"
        return result

    if not finite:
        result["reason"] = "non_finite_csi"
        return result

    if cycles < minimum_cycles:
        result["reason"] = "insufficient_tdm_cycles"
        return result

    result["accepted"] = True
    result["reason"] = "accepted"

    return result


def update_view_manifest(view_dir, qc_result):
    view_dir = Path(view_dir)

    manifest_file = (
        view_dir / "view_manifest.json"
    )

    with open(
        manifest_file,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    manifest["quality"] = qc_result

    manifest["state"] = (
        "accepted"
        if qc_result["accepted"]
        else "rejected"
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )


def update_session_manifest(
    session_dir,
    view_id,
    qc_result,
):
    session_dir = Path(session_dir)

    manifest_file = (
        session_dir / "manifest.json"
    )

    with open(
        manifest_file,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    state = (
        "accepted"
        if qc_result["accepted"]
        else "rejected"
    )

    for item in manifest["views"]:
        if item["view_id"] == view_id:
            item["state"] = state
            break

    passed = sum(
        1
        for item in manifest["views"]
        if item["state"] == "accepted"
    )

    failed = sum(
        1
        for item in manifest["views"]
        if item["state"] == "rejected"
    )

    manifest["counters"]["views_passed"] = passed
    manifest["counters"]["views_failed"] = failed

    manifest["status"] = (
        "captured_with_accepted_view"
        if qc_result["accepted"]
        else "captured_with_rejected_view"
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )
