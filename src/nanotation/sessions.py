from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .annotations import DEFAULT_PATH_SMOOTHNESS
from .mrc_io import MrcFrameRecord


SESSION_FORMAT = "nanotation-session"
SESSION_VERSION = 3


@dataclass(frozen=True, slots=True)
class NanotationSession:
    source_folder: Path
    annotations: np.ndarray
    frame_count: int
    first_filename: str
    last_filename: str
    frame_number: int
    zoom: float
    camera_3d: dict[str, object]
    path_smoothness: float


def write_session_file(
    path: Path,
    *,
    source_folder: Path,
    annotations: np.ndarray,
    records: Sequence[MrcFrameRecord],
    frame_number: int,
    zoom: float,
    camera_3d: dict[str, object],
    path_smoothness: float = DEFAULT_PATH_SMOOTHNESS,
) -> None:
    if not records:
        raise ValueError("Load a time-series before saving a session.")
    points = np.asarray(annotations, dtype=float)
    if points.size == 0:
        points = np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Session annotations must be finite N x 3 coordinates.")

    payload = {
        "format": SESSION_FORMAT,
        "version": SESSION_VERSION,
        "source_folder": str(source_folder.expanduser().resolve()),
        "annotations_zyx": points.tolist(),
        "image_count": len(records),
        "first_filename": records[0].name,
        "last_filename": records[-1].name,
        "frame_number": int(frame_number),
        "zoom": float(zoom),
        "camera_3d": camera_3d,
        "path_smoothness": float(path_smoothness),
    }
    if payload["frame_number"] < 1 or payload["frame_number"] > len(records):
        raise ValueError("Session frame number is outside the loaded time-series.")
    if not np.isfinite(payload["zoom"]) or payload["zoom"] <= 0:
        raise ValueError("Session zoom must be a positive finite value.")
    if (
        not np.isfinite(payload["path_smoothness"])
        or payload["path_smoothness"] < 0
        or payload["path_smoothness"] > 1
    ):
        raise ValueError("Session path smoothness must be between 0 and 1.")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_session_file(path: Path) -> NanotationSession:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read session file: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Session file must contain a JSON object.")
    version = payload.get("version")
    if payload.get("format") != SESSION_FORMAT or version not in (1, 2, SESSION_VERSION):
        raise ValueError("Unsupported Nanotation session format or version.")

    try:
        points = np.asarray(payload["annotations_zyx"], dtype=float)
        if points.size == 0:
            points = np.empty((0, 3), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("invalid annotation coordinates")
        frame_count = int(payload["image_count"])
        frame_number = (
            int(payload["slice_index"]) + 1
            if version == 1
            else int(payload["frame_number"])
        )
        zoom = float(payload["zoom"])
        path_smoothness = (
            float(payload["path_smoothness"])
            if version == SESSION_VERSION
            else DEFAULT_PATH_SMOOTHNESS
        )
        if (
            frame_count < 1
            or frame_number < 1
            or frame_number > frame_count
            or not np.isfinite(zoom)
            or zoom <= 0
            or not np.isfinite(path_smoothness)
            or path_smoothness < 0
            or path_smoothness > 1
        ):
            raise ValueError("invalid numeric session value")
        camera_3d = payload.get("camera_3d", {})
        if not isinstance(camera_3d, dict):
            raise ValueError("invalid 3D camera state")
        return NanotationSession(
            source_folder=Path(payload["source_folder"]).expanduser(),
            annotations=points,
            frame_count=frame_count,
            first_filename=str(payload["first_filename"]),
            last_filename=str(payload["last_filename"]),
            frame_number=frame_number,
            zoom=zoom,
            camera_3d=camera_3d,
            path_smoothness=path_smoothness,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Nanotation session: {exc}") from exc
