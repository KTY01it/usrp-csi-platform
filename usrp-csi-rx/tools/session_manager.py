#!/usr/bin/env python3

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


class SessionManager:

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sessions_root = self.root / "sessions"

    def _now(self):
        return datetime.now().astimezone()

    def _timestamp(self):
        return self._now().strftime("%Y%m%d_%H%M%S")

    def _iso_time(self):
        return self._now().isoformat()

    def _copy_config(self, src, dst):
        if not src.exists():
            raise FileNotFoundError(src)

        shutil.copy2(src, dst)

    def create_session(
        self,
        name,
        mode="step",
        trajectory="manual_step",
    ):
        if not name:
            raise ValueError("Session name cannot be empty")

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in name
        )

        session_id = (
            f"{safe_name}_{self._timestamp()}"
        )

        session_dir = (
            self.sessions_root
            / safe_name
            / session_id
        )

        if session_dir.exists():
            raise RuntimeError(
                f"Session already exists: {session_dir}"
            )

        views_dir = session_dir / "views"

        views_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        config_dir = (
            self.root / "configs"
        )

        snapshot_dir = (
            session_dir / "config_snapshot"
        )

        snapshot_dir.mkdir()

        config_files = {
            "system.yaml":
                config_dir / "platform/system.yaml",

            "geometry.yaml":
                config_dir / "platform/geometry.yaml",

            "acquisition.yaml":
                config_dir / "platform/acquisition.yaml",

            "calibration.yaml":
                config_dir / "platform/calibration.yaml",

            "trajectory.yaml":
                config_dir
                / "trajectories"
                / f"{trajectory}.yaml",
        }

        for dst_name, src in config_files.items():
            self._copy_config(
                src,
                snapshot_dir / dst_name
            )

        manifest = {
            "schema_version": 1,

            "session_id": session_id,
            "session_name": safe_name,

            "created_at": self._iso_time(),

            "mode": mode,
            "trajectory": trajectory,

            "status": "prepared",

            "platform": {
                "role": "rx_controller",
                "framework": "csi-sense",
            },

            "views": [],

            "counters": {
                "views_total": 0,
                "views_passed": 0,
                "views_failed": 0,
            },

            "scientific_status": {
                "absolute_aoa": False,
                "absolute_aod": False,
                "physical_tof": False,
                "radar_grade_range": False,
            },
        }

        manifest_file = (
            session_dir / "manifest.json"
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

        self.create_view(
            session_dir,
            view_index=0,
            state="prepared",
        )

        return session_dir

    def create_view(
        self,
        session_dir,
        view_index,
        state="prepared",
    ):
        session_dir = Path(session_dir)

        view_id = f"view_{view_index:03d}"

        view_dir = (
            session_dir
            / "views"
            / view_id
        )

        view_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        for subdir in [
            "raw",
            "csi",
            "features",
            "quality",
            "logs",
        ]:
            (
                view_dir / subdir
            ).mkdir()

        view_manifest = {
            "schema_version": 1,

            "view_id": view_id,

            "created_at":
                self._iso_time(),

            "state": state,

            "pose": {
                "xyz_m": None,
                "yaw_deg": None,
                "pitch_deg": None,
                "roll_deg": None,
                "measured": False,
            },

            "capture": {
                "started_at": None,
                "finished_at": None,
                "duration_s": None,
            },

            "quality": {
                "accepted": None,
                "reason": None,
            },

            "artifacts": {},
        }

        with open(
            view_dir / "view_manifest.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                view_manifest,
                f,
                indent=2,
                ensure_ascii=False
            )

        manifest_file = (
            session_dir / "manifest.json"
        )

        with open(
            manifest_file,
            "r",
            encoding="utf-8"
        ) as f:
            manifest = json.load(f)

        manifest["views"].append({
            "view_id": view_id,
            "path": f"views/{view_id}",
            "state": state,
        })

        manifest[
            "counters"
        ]["views_total"] = len(
            manifest["views"]
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

        return view_dir
