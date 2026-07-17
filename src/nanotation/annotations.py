from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline

from .mrc_io import MrcFrameRecord


CSV_FIELDNAMES = ("filename", "frame_number", "x", "y")
DEFAULT_PATH_SMOOTHNESS = 0.15
SMOOTHNESS_MAX_NORMALIZED_RMS = 0.02


class SmoothedPath:
    """Frame-indexed smoothing spline fitted to noisy checkpoint measurements."""

    def __init__(
        self,
        coordinates: np.ndarray,
        smoothness: float = DEFAULT_PATH_SMOOTHNESS,
        image_shape: tuple[int, int] | None = None,
    ) -> None:
        self.smoothness = _validated_smoothness(smoothness)
        self._frames = np.empty(0, dtype=float)
        self._y_values = np.empty(0, dtype=float)
        self._x_values = np.empty(0, dtype=float)
        self._y_spline = None
        self._x_spline = None
        self._frame_origin = 0.0
        self._frame_scale = 1.0
        self._y_origin = 0.0
        self._x_origin = 0.0
        self._y_scale = 1.0
        self._x_scale = 1.0

        points = np.asarray(coordinates, dtype=float)
        if points.size == 0 or points.ndim != 2 or points.shape[1] != 3:
            return
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 2:
            return

        frame_indices = np.rint(points[:, 0]).astype(int)
        unique_frames, inverse = np.unique(frame_indices, return_inverse=True)
        if len(unique_frames) < 2:
            return

        counts = np.bincount(inverse).astype(float)
        self._frames = unique_frames.astype(float)
        self._y_values = np.bincount(inverse, weights=points[:, 1]) / counts
        self._x_values = np.bincount(inverse, weights=points[:, 2]) / counts
        self._frame_origin = float(self._frames[0])
        self._frame_scale = max(float(self._frames[-1] - self._frames[0]), 1.0)
        self._y_origin = float(self._y_values.mean())
        self._x_origin = float(self._x_values.mean())

        if image_shape is None:
            self._y_scale = max(float(np.ptp(self._y_values)), 1.0)
            self._x_scale = max(float(np.ptp(self._x_values)), 1.0)
        else:
            height, width = image_shape
            self._y_scale = max(float(height - 1), 1.0)
            self._x_scale = max(float(width - 1), 1.0)

        if len(self._frames) >= 3:
            normalized_frames = self._normalize_frames(self._frames)
            residual_rms = self.smoothness * SMOOTHNESS_MAX_NORMALIZED_RMS
            residual_budget = len(self._frames) * residual_rms**2
            spline_degree = min(3, len(self._frames) - 1)
            self._y_spline = UnivariateSpline(
                normalized_frames,
                (self._y_values - self._y_origin) / self._y_scale,
                k=spline_degree,
                s=residual_budget,
                ext=2,
            )
            self._x_spline = UnivariateSpline(
                normalized_frames,
                (self._x_values - self._x_origin) / self._x_scale,
                k=spline_degree,
                s=residual_budget,
                ext=2,
            )

    @property
    def is_valid(self) -> bool:
        return len(self._frames) >= 2

    def at_frame(self, frame_index: int | float) -> np.ndarray | None:
        frame = float(frame_index)
        if not self.is_valid or frame < self._frames[0] or frame > self._frames[-1]:
            return None
        if self._y_spline is None or self._x_spline is None:
            y_value = np.interp(frame, self._frames, self._y_values)
            x_value = np.interp(frame, self._frames, self._x_values)
        else:
            normalized_frame = self._normalize_frames(frame)
            y_value = self._y_origin + self._y_scale * float(self._y_spline(normalized_frame))
            x_value = self._x_origin + self._x_scale * float(self._x_spline(normalized_frame))
        return np.array([frame, y_value, x_value], dtype=float)

    def sample(self, *, samples_per_interval: int = 16, max_samples: int = 4096) -> np.ndarray:
        if not self.is_valid:
            return np.empty((0, 3), dtype=float)
        interval_count = len(self._frames) - 1
        sample_count = min(max_samples, max(2, interval_count * samples_per_interval + 1))
        frames = np.linspace(self._frames[0], self._frames[-1], sample_count)
        return np.asarray([self.at_frame(frame) for frame in frames], dtype=float)

    def _normalize_frames(self, frames):
        return (frames - self._frame_origin) / self._frame_scale


def _validated_smoothness(smoothness: float) -> float:
    value = float(smoothness)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError("Path smoothness must be between 0 and 1.")
    return value


def annotation_rows(
    coordinates: np.ndarray,
    records: Sequence[MrcFrameRecord],
    smoothness: float = DEFAULT_PATH_SMOOTHNESS,
) -> list[dict[str, object]]:
    """Convert checkpoint and intersection coordinates into frame-wise export rows."""

    points = np.asarray(coordinates, dtype=float)
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Annotation coordinates must be an N x 3 array in z-y-x order.")
    if not np.isfinite(points).all():
        raise ValueError("Annotation coordinates must be finite.")

    point_frame_indices = np.rint(points[:, 0]).astype(int)
    for point_id, frame_index in enumerate(point_frame_indices, start=1):
        if frame_index < 0 or frame_index >= len(records):
            z_value = points[point_id - 1, 0]
            raise ValueError(f"Point {point_id} is outside the loaded volume (z={z_value:g}).")

    frame_indices = set(int(frame_index) for frame_index in point_frame_indices)
    if len(points) >= 2:
        first_frame = max(0, int(np.ceil(points[:, 0].min())))
        last_frame = min(len(records) - 1, int(np.floor(points[:, 0].max())))
        frame_indices.update(range(first_frame, last_frame + 1))

    path = SmoothedPath(points, smoothness, records[0].shape)
    rows: list[dict[str, object]] = []
    for frame_index in sorted(frame_indices):
        coordinate = path.at_frame(frame_index)
        if coordinate is None:
            coordinate = points[point_frame_indices == frame_index].mean(axis=0)
            coordinate[0] = frame_index
        _z_value, y_value, x_value = coordinate
        rows.append(
            {
                "filename": records[frame_index].name,
                "frame_number": frame_index + 1,
                "x": float(x_value),
                "y": float(y_value),
            }
        )
    return rows


def write_annotations_csv(
    path: Path,
    coordinates: np.ndarray,
    records: Sequence[MrcFrameRecord],
    smoothness: float = DEFAULT_PATH_SMOOTHNESS,
) -> int:
    rows = annotation_rows(coordinates, records, smoothness)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def annotation_xyz_coordinates(coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=float)
    if points.size == 0 or points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    return points[:, [2, 1, 0]]


def frame_neighbor_edges(xyz_coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz_coordinates, dtype=float)
    if points.size == 0 or points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        return np.empty((0, 2, 3), dtype=float)
    order = np.argsort(points[:, 2], kind="stable")
    ordered = points[order]
    return np.stack((ordered[:-1], ordered[1:]), axis=1)


def dashed_neighbor_segments(
    xyz_coordinates: np.ndarray,
    *,
    dash_count: int = 10,
    dash_fraction: float = 0.55,
) -> np.ndarray:
    edges = frame_neighbor_edges(xyz_coordinates)
    if len(edges) == 0:
        return np.empty((0, 3), dtype=np.float32)

    segments = []
    for start, end in edges:
        direction = end - start
        for dash_index in range(dash_count):
            start_fraction = dash_index / dash_count
            end_fraction = (dash_index + dash_fraction) / dash_count
            segments.extend(
                (
                    start + direction * start_fraction,
                    start + direction * end_fraction,
                )
            )
    return np.asarray(segments, dtype=np.float32)


def path_intersection_at_frame(
    zyx_coordinates: np.ndarray,
    frame_index: int,
    smoothness: float = DEFAULT_PATH_SMOOTHNESS,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    return SmoothedPath(zyx_coordinates, smoothness, image_shape).at_frame(frame_index)
