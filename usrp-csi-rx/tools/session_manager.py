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

            "tx_pose": {
                "xyz_m": None,
                "yaw_deg": None,
                "pitch_deg": None,
                "roll_deg": None,
                "measured": False,
            },

            "rx_pose": {
                "xyz_m": None,
                "yaw_deg": None,
                "pitch_deg": None,
                "roll_deg": None,
                "measured": False,
            },

            "pose_registration": {
                "method": "manual_known_pose",
                "coordinate_system": "room_local",
                "valid_for_spatial_fusion": False,
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

    def set_view_pose(
        self,
        view_dir,
        tx_xyz_m,
        rx_xyz_m,
        tx_yaw_deg=0.0,
        tx_pitch_deg=0.0,
        tx_roll_deg=0.0,
        rx_yaw_deg=0.0,
        rx_pitch_deg=0.0,
        rx_roll_deg=0.0,
    ):
        view_dir = Path(
            view_dir
        ).resolve()

        manifest_file = (
            view_dir
            / "view_manifest.json"
        )

        if not manifest_file.exists():
            raise FileNotFoundError(
                manifest_file
            )

        def validate_xyz(
            value,
            name,
        ):
            if (
                not isinstance(
                    value,
                    (list, tuple)
                )
                or len(value) != 3
            ):
                raise ValueError(
                    f"{name} must contain [x,y,z]"
                )

            return [
                float(x)
                for x in value
            ]

        tx_xyz = validate_xyz(
            tx_xyz_m,
            "tx_xyz_m",
        )

        rx_xyz = validate_xyz(
            rx_xyz_m,
            "rx_xyz_m",
        )

        with open(
            manifest_file,
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        manifest["tx_pose"] = {
            "xyz_m": tx_xyz,
            "yaw_deg": float(
                tx_yaw_deg
            ),
            "pitch_deg": float(
                tx_pitch_deg
            ),
            "roll_deg": float(
                tx_roll_deg
            ),
            "measured": True,
        }

        manifest["rx_pose"] = {
            "xyz_m": rx_xyz,
            "yaw_deg": float(
                rx_yaw_deg
            ),
            "pitch_deg": float(
                rx_pitch_deg
            ),
            "roll_deg": float(
                rx_roll_deg
            ),
            "measured": True,
        }

        manifest[
            "pose_registration"
        ] = {
            "method":
                "manual_known_pose",

            "coordinate_system":
                "room_local",

            "valid_for_spatial_fusion":
                True,
        }

        with open(
            manifest_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return manifest


    def create_next_view(
        self,
        session_dir,
        state="prepared",
    ):
        session_dir = Path(
            session_dir
        ).resolve()

        manifest_file = (
            session_dir
            / "manifest.json"
        )

        if not manifest_file.exists():
            raise FileNotFoundError(
                manifest_file
            )

        with open(
            manifest_file,
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        existing = manifest.get(
            "views",
            [],
        )

        indices = []

        for item in existing:
            view_id = item.get(
                "view_id",
                "",
            )

            if not view_id.startswith(
                "view_"
            ):
                continue

            try:
                indices.append(
                    int(
                        view_id.split(
                            "_"
                        )[1]
                    )
                )
            except Exception:
                continue

        next_index = (
            max(indices) + 1
            if indices
            else 0
        )

        return self.create_view(
            session_dir,
            view_index=next_index,
            state=state,
        )

    def remove_prepared_view(
        self,
        session_dir,
        view_dir,
    ):
        session_dir = Path(
            session_dir
        ).resolve()

        view_dir = Path(
            view_dir
        ).resolve()

        manifest_file = (
            session_dir
            / "manifest.json"
        )

        view_manifest_file = (
            view_dir
            / "view_manifest.json"
        )

        if not manifest_file.exists():
            raise FileNotFoundError(
                manifest_file
            )

        if not view_manifest_file.exists():
            raise FileNotFoundError(
                view_manifest_file
            )

        with open(
            view_manifest_file,
            "r",
            encoding="utf-8",
        ) as f:
            vm = json.load(f)

        if vm.get("state") != "prepared":
            raise RuntimeError(
                "Only an unconsumed prepared view "
                "may be removed"
            )

        with open(
            manifest_file,
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        view_id = view_dir.name

        manifest["views"] = [
            item
            for item in manifest.get(
                "views",
                []
            )
            if item.get("view_id") != view_id
        ]

        manifest[
            "counters"
        ][
            "views_total"
        ] = len(
            manifest["views"]
        )

        shutil.rmtree(
            view_dir
        )

        with open(
            manifest_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2,
                ensure_ascii=False,
            )
