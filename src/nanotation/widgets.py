from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qtpy.QtCore import QEvent, QTimer, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .mrc_io import LazyMrcVolume, MrcSliceRecord, scan_mrc_folder, volume_scale


VOLUME_LAYER_NAME = "MRC volume"
POINTS_LAYER_NAME = "Path checkpoints"
INTERSECTION_LAYER_NAME = "Path intersection"
POINT_BORDER_COLOR = "#0055ffff"
POINT_OPACITY = 0.6
POINT_SIZE = 32
DEFAULT_OUTPUT_IMAGE_SIZE = int(POINT_SIZE * 1.5)
INTERSECTION_COLOR = "#ffffffff"
INTERSECTION_OPACITY = 0.4
INTERSECTION_SIZE = 24
SESSION_FORMAT = "nanotation-session"
SESSION_VERSION = 1
ZOOM_STEP = 1.25
MIN_ZOOM = 0.01
MAX_ZOOM = 100.0
DEFAULT_INTERPOLATION = "bicubic"
EMAN2_IMAGE_ORIENTATION_2D = ("up", "right")
HISTOGRAM_BINS = 96
CONTRAST_STD_MULTIPLIER = 5.0


def annotation_rows(
    coordinates: np.ndarray,
    records: Sequence[MrcSliceRecord],
    coordinate_scale: float = 1.0,
) -> list[dict[str, object]]:
    """Convert checkpoint and intersection coordinates into slice-wise export rows."""

    scale = _validated_coordinate_scale(coordinate_scale)
    points = np.asarray(coordinates, dtype=float)
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Annotation coordinates must be an N x 3 array in z-y-x order.")
    if not np.isfinite(points).all():
        raise ValueError("Annotation coordinates must be finite.")

    point_slice_indices = np.rint(points[:, 0]).astype(int)
    for point_id, slice_index in enumerate(point_slice_indices, start=1):
        if slice_index < 0 or slice_index >= len(records):
            z_value = points[point_id - 1, 0]
            raise ValueError(f"Point {point_id} is outside the loaded volume (z={z_value:g}).")

    slice_indices = set(int(slice_index) for slice_index in point_slice_indices)
    if len(points) >= 2:
        first_slice = max(0, int(np.ceil(points[:, 0].min())))
        last_slice = min(len(records) - 1, int(np.floor(points[:, 0].max())))
        slice_indices.update(range(first_slice, last_slice + 1))

    rows: list[dict[str, object]] = []
    for slice_index in sorted(slice_indices):
        coordinate = path_intersection_at_slice(points, slice_index)
        if coordinate is None:
            coordinate = points[point_slice_indices == slice_index].mean(axis=0)
            coordinate[0] = slice_index
        _z_value, y_value, x_value = coordinate
        rows.append(
            {
                "filename": records[slice_index].name,
                "slice_index": slice_index,
                "x": float(x_value),
                "y": float(y_value),
                "xsc": float(scale * x_value),
                "ysc": float(scale * y_value),
            }
        )
    return rows


def write_annotations_csv(
    path: Path,
    coordinates: np.ndarray,
    records: Sequence[MrcSliceRecord],
    coordinate_scale: float = 1.0,
) -> int:
    rows = annotation_rows(coordinates, records, coordinate_scale=coordinate_scale)
    fieldnames = ["filename", "slice_index", "x", "y", "xsc", "ysc"]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _validated_coordinate_scale(coordinate_scale: float) -> float:
    scale = float(coordinate_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Coordinate scale must be a positive finite value.")
    return scale


def finite_intensity_range(values: np.ndarray) -> tuple[float, float] | None:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def finite_intensity_standard_deviation(values: np.ndarray) -> float | None:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    standard_deviation = float(finite.std())
    return standard_deviation if np.isfinite(standard_deviation) else None


def point_slice_summary(coordinates: np.ndarray, records: Sequence[MrcSliceRecord]) -> str:
    points = np.asarray(coordinates, dtype=float)
    if points.size == 0:
        return "Annotated slices: none"
    if points.ndim != 2 or points.shape[1] != 3:
        return "Annotated slices: invalid point data"

    counts: dict[int, int] = {}
    for z_value in points[:, 0]:
        slice_index = int(np.rint(z_value))
        if 0 <= slice_index < len(records):
            counts[slice_index] = counts.get(slice_index, 0) + 1

    if not counts:
        return "Annotated slices: none inside loaded volume"

    parts = []
    for slice_index, count in sorted(counts.items()):
        noun = "point" if count == 1 else "points"
        parts.append(f"{slice_index} ({records[slice_index].name}): {count} {noun}")
    return "Annotated slices: " + "; ".join(parts)


def annotation_xyz_coordinates(coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=float)
    if points.size == 0 or points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    return points[:, [2, 1, 0]]


def volume_box_segments(image_count: int, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    x_extent = max(1, width - 1)
    y_extent = max(1, height - 1)
    z_extent = max(1, image_count - 1)
    corners = np.array(
        [
            [0, 0, 0],
            [x_extent, 0, 0],
            [x_extent, y_extent, 0],
            [0, y_extent, 0],
            [0, 0, z_extent],
            [x_extent, 0, z_extent],
            [x_extent, y_extent, z_extent],
            [0, y_extent, z_extent],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    return corners[edges.ravel()]


def nearest_projected_point(
    projected_positions: np.ndarray,
    click_position: tuple[float, float],
    max_distance: float = 24.0,
) -> int | None:
    positions = np.asarray(projected_positions, dtype=float)
    if positions.size == 0:
        return None
    distances = np.linalg.norm(positions[:, :2] - np.asarray(click_position), axis=1)
    nearest = int(np.argmin(distances))
    return nearest if distances[nearest] <= max_distance else None


def homogeneous_canvas_positions(projected_positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(projected_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("Projected positions must be an N x 2-or-more array.")
    if positions.shape[1] < 4:
        return positions[:, :2]
    weights = positions[:, 3]
    result = np.full((len(positions), 2), np.inf, dtype=float)
    valid = np.isfinite(weights) & (np.abs(weights) > np.finfo(float).eps)
    result[valid] = positions[valid, :2] / weights[valid, None]
    return result


def slice_neighbor_edges(xyz_coordinates: np.ndarray) -> np.ndarray:
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
    edges = slice_neighbor_edges(xyz_coordinates)
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


def path_intersection_at_slice(
    zyx_coordinates: np.ndarray,
    slice_index: int,
) -> np.ndarray | None:
    points = np.asarray(zyx_coordinates, dtype=float)
    if points.size == 0 or points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        return None
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return None

    ordered = points[np.argsort(points[:, 0], kind="stable")]
    z_value = float(slice_index)
    if z_value < ordered[0, 0] or z_value > ordered[-1, 0]:
        return None

    exact = np.flatnonzero(np.isclose(ordered[:, 0], z_value))
    if exact.size:
        intersection = ordered[exact].mean(axis=0)
        intersection[0] = z_value
        return intersection

    upper_index = int(np.searchsorted(ordered[:, 0], z_value, side="right"))
    lower = ordered[upper_index - 1]
    upper = ordered[upper_index]
    fraction = (z_value - lower[0]) / (upper[0] - lower[0])
    intersection = lower + (upper - lower) * fraction
    intersection[0] = z_value
    return intersection


@dataclass(frozen=True)
class NanotationSession:
    source_folder: Path
    annotations: np.ndarray
    coordinate_scale: float
    image_count: int
    first_filename: str
    last_filename: str
    slice_index: int
    zoom: float
    camera_3d: dict[str, object]


def write_session_file(
    path: Path,
    *,
    source_folder: Path,
    annotations: np.ndarray,
    coordinate_scale: float,
    records: Sequence[MrcSliceRecord],
    slice_index: int,
    zoom: float,
    camera_3d: dict[str, object],
) -> None:
    if not records:
        raise ValueError("Load a volume before saving a session.")
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
        "coordinate_scale": _validated_coordinate_scale(coordinate_scale),
        "image_count": len(records),
        "first_filename": records[0].name,
        "last_filename": records[-1].name,
        "slice_index": int(slice_index),
        "zoom": float(zoom),
        "camera_3d": camera_3d,
    }
    if not np.isfinite(payload["zoom"]) or payload["zoom"] <= 0:
        raise ValueError("Session zoom must be a positive finite value.")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_session_file(path: Path) -> NanotationSession:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read session file: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Session file must contain a JSON object.")
    if payload.get("format") != SESSION_FORMAT or payload.get("version") != SESSION_VERSION:
        raise ValueError("Unsupported Nanotation session format or version.")

    try:
        points = np.asarray(payload["annotations_zyx"], dtype=float)
        if points.size == 0:
            points = np.empty((0, 3), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("invalid annotation coordinates")
        coordinate_scale = _validated_coordinate_scale(payload["coordinate_scale"])
        image_count = int(payload["image_count"])
        slice_index = int(payload["slice_index"])
        zoom = float(payload["zoom"])
        if image_count < 1 or slice_index < 0 or not np.isfinite(zoom) or zoom <= 0:
            raise ValueError("invalid numeric session value")
        camera_3d = payload.get("camera_3d", {})
        if not isinstance(camera_3d, dict):
            raise ValueError("invalid 3D camera state")
        return NanotationSession(
            source_folder=Path(payload["source_folder"]).expanduser(),
            annotations=points,
            coordinate_scale=coordinate_scale,
            image_count=image_count,
            first_filename=str(payload["first_filename"]),
            last_filename=str(payload["last_filename"]),
            slice_index=slice_index,
            zoom=zoom,
            camera_3d=camera_3d,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Nanotation session: {exc}") from exc


class Annotation3DPlot(QWidget):
    """Read-only, rotatable 3D view of annotation coordinates and path."""

    slice_requested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        from vispy import scene

        self._scene = scene
        self._volume_shape: tuple[int, int, int] | None = None
        self._xyz = np.empty((0, 3), dtype=np.float32)
        self._mouse_press_position: tuple[float, float] | None = None
        self._selection_handled_on_press = False
        self._canvas = scene.SceneCanvas(
            keys=None,
            show=False,
            size=(400, 400),
            bgcolor="#202124",
        )
        self._view = self._canvas.central_widget.add_view()
        self._camera = scene.cameras.TurntableCamera(fov=45, elevation=25, azimuth=35)
        self._view.camera = self._camera
        self._bounding_box = scene.visuals.Line(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.9),
            width=1.5,
            connect="segments",
            parent=self._view.scene,
        )
        self._slice_outline = scene.visuals.Line(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.3, 0.85, 1.0, 0.9),
            width=1.5,
            connect="strip",
            parent=self._view.scene,
        )
        self._line = scene.visuals.Line(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.35, 0.7, 1.0, 0.9),
            width=2,
            connect="segments",
            parent=self._view.scene,
        )
        self._markers = scene.visuals.Markers(parent=self._view.scene)
        self._markers.set_data(
            np.empty((0, 3), dtype=np.float32),
            size=14,
            face_color=(0.0, 0.333, 1.0, 0.75),
            edge_color=(0.5, 0.8, 1.0, 1.0),
            edge_width=1,
        )
        self._axis = scene.visuals.XYZAxis(parent=self._view.scene)

        instructions = QLabel("3D annotations — click a point to jump; drag to rotate; scroll to zoom")
        instructions.setWordWrap(True)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(instructions)
        layout.addWidget(self._canvas.native, stretch=1)
        self.setLayout(layout)
        self.setMinimumHeight(180)
        self._canvas.native.setMinimumHeight(150)
        self._slice_label = QLabel("Slice index: —", self._canvas.native)
        self._slice_label.setStyleSheet(
            "color: white; background-color: rgba(0, 0, 0, 150); padding: 3px 6px;"
        )
        self._slice_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._slice_label.move(8, 8)
        self._slice_label.adjustSize()
        self._slice_label.raise_()
        self.setToolTip("Read-only view; points and connecting line follow annotation order.")
        self._canvas.native.installEventFilter(self)
        self._canvas.events.mouse_press.connect(self._on_canvas_mouse_press, position="first")
        self._markers.interactive = True
        self._markers.events.mouse_press.connect(self._on_marker_mouse_press)

    def set_volume_shape(self, image_count: int, image_shape: tuple[int, int]) -> None:
        height, width = image_shape
        volume_shape = (int(image_count), int(height), int(width))
        if volume_shape == self._volume_shape:
            return
        self._volume_shape = volume_shape

        x_extent = max(1, width - 1)
        y_extent = max(1, height - 1)
        z_extent = max(1, image_count - 1)
        self._camera.set_range(
            x=(0, x_extent),
            y=(0, y_extent),
            z=(0, z_extent),
            margin=0.08,
        )
        self._bounding_box.set_data(
            pos=volume_box_segments(image_count, image_shape),
            connect="segments",
        )
        axis_size = max(x_extent, y_extent, z_extent) * 0.12
        self._axis.transform = self._scene.transforms.STTransform(
            scale=(axis_size, axis_size, axis_size)
        )
        self.set_slice_index(0)

    def set_slice_index(self, slice_index: int) -> None:
        if self._volume_shape is None:
            return
        image_count, height, width = self._volume_shape
        index = min(max(0, int(slice_index)), max(0, image_count - 1))
        self._slice_label.setText(f"Slice index: {index}")
        self._slice_label.adjustSize()

        x_extent = max(1, width - 1)
        y_extent = max(1, height - 1)
        z_value = float(index)
        outline = np.array(
            [
                [0, 0, z_value],
                [x_extent, 0, z_value],
                [x_extent, y_extent, z_value],
                [0, y_extent, z_value],
                [0, 0, z_value],
            ],
            dtype=np.float32,
        )
        self._slice_outline.set_data(pos=outline, connect="strip")
        self._canvas.update()

    def set_annotations(self, coordinates: np.ndarray) -> None:
        xyz = annotation_xyz_coordinates(coordinates).astype(np.float32, copy=False)
        self._xyz = xyz
        self._markers.set_data(
            xyz,
            size=14,
            face_color=(0.0, 0.333, 1.0, 0.75),
            edge_color=(0.5, 0.8, 1.0, 1.0),
            edge_width=1,
        )
        self._line.set_data(pos=dashed_neighbor_segments(xyz), connect="segments")
        self._canvas.update()

    def camera_state(self) -> dict[str, object]:
        return {
            "azimuth": float(self._camera.azimuth),
            "elevation": float(self._camera.elevation),
            "roll": float(self._camera.roll),
            "scale_factor": float(self._camera.scale_factor),
            "center": [float(value) for value in self._camera.center],
        }

    def restore_camera_state(self, state: dict[str, object]) -> None:
        try:
            center = np.asarray(state["center"], dtype=float)
            values = {
                "azimuth": float(state["azimuth"]),
                "elevation": float(state["elevation"]),
                "roll": float(state["roll"]),
                "scale_factor": float(state["scale_factor"]),
            }
            if center.shape != (3,) or not np.isfinite(center).all():
                return
            if not all(np.isfinite(value) for value in values.values()):
                return
            self._camera.center = tuple(center)
            for name, value in values.items():
                setattr(self._camera, name, value)
            self._canvas.update()
        except (KeyError, TypeError, ValueError):
            return

    def eventFilter(self, watched, event) -> bool:
        if watched is self._canvas.native and event.type() in (
            QEvent.MouseButtonPress,
            QEvent.MouseButtonRelease,
        ):
            if event.button() != Qt.LeftButton:
                return False
            position = event.position() if hasattr(event, "position") else event.localPos()
            canvas_position = (float(position.x()), float(position.y()))
            if event.type() == QEvent.MouseButtonPress:
                self._mouse_press_position = canvas_position
                self._selection_handled_on_press = False
            else:
                if not self._selection_handled_on_press:
                    self._handle_canvas_click(canvas_position)
                else:
                    self._mouse_press_position = None
        return super().eventFilter(watched, event)

    def _on_canvas_mouse_press(self, event) -> None:
        if event.button != 1:
            return
        position = tuple(float(value) for value in event.pos[:2])
        if self._select_at_canvas_position(position):
            self._selection_handled_on_press = True
            event.blocked = True

    def _on_marker_mouse_press(self, event) -> None:
        if event.button != 1:
            return
        position = tuple(float(value) for value in event.mouse_event.pos[:2])
        if self._select_at_canvas_position(position, require_distance=False):
            self._selection_handled_on_press = True
            event.handled = True

    def _handle_canvas_click(self, release_position: tuple[float, float]) -> None:
        if self._mouse_press_position is None:
            return
        press_position = self._mouse_press_position
        self._mouse_press_position = None
        if np.linalg.norm(np.subtract(release_position, press_position)) > 4:
            return
        if self._xyz.size == 0:
            return

        self._select_at_canvas_position(release_position)

    def _select_at_canvas_position(
        self,
        position: tuple[float, float],
        *,
        require_distance: bool = True,
    ) -> bool:
        if self._xyz.size == 0:
            return False
        transform = self._markers.get_transform(map_from="visual", map_to="canvas")
        projected = homogeneous_canvas_positions(transform.map(self._xyz))
        max_distance = 24.0 if require_distance else float("inf")
        point_index = nearest_projected_point(projected, position, max_distance=max_distance)
        if point_index is None:
            return False
        slice_index = int(np.rint(self._xyz[point_index, 2]))
        self.slice_requested.emit(slice_index)
        self._slice_label.setText(f"Slice index: {slice_index} — selected from 3D")
        self._slice_label.adjustSize()
        return True


class HistogramWidget(QWidget):
    """Compact histogram for the currently displayed image slice."""

    contrast_changed = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._counts = np.empty(0, dtype=float)
        self._edges = np.empty(0, dtype=float)
        self._contrast_low: float | None = None
        self._contrast_high: float | None = None
        self._dragging_threshold: str | None = None
        self.setMinimumHeight(90)
        self.setMouseTracking(True)

    def set_histogram(
        self,
        values: np.ndarray | None,
        *,
        contrast_low: float | None = None,
        contrast_high: float | None = None,
    ) -> None:
        self._contrast_low = contrast_low
        self._contrast_high = contrast_high
        if values is None:
            self._counts = np.empty(0, dtype=float)
            self._edges = np.empty(0, dtype=float)
            self.update()
            return

        data = np.asarray(values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            self._counts = np.empty(0, dtype=float)
            self._edges = np.empty(0, dtype=float)
            self.update()
            return

        data_min = float(data.min())
        data_max = float(data.max())
        if data_min == data_max:
            padding = max(abs(data_min) * 0.001, 0.5)
            data_min -= padding
            data_max += padding
        counts, edges = np.histogram(data, bins=HISTOGRAM_BINS, range=(data_min, data_max))
        self._counts = counts.astype(float)
        self._edges = edges.astype(float)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#202124"))

        plot_rect = self._plot_rect()
        painter.setPen(QPen(QColor("#5f6368"), 1))
        painter.drawRect(plot_rect)

        if self._counts.size == 0 or self._edges.size < 2:
            painter.setPen(QColor("#c7c7c7"))
            painter.drawText(plot_rect, Qt.AlignCenter, "No frame histogram")
            painter.end()
            return

        maximum = float(self._counts.max())
        if maximum > 0:
            bar_width = plot_rect.width() / self._counts.size
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#9aa0a6"))
            for index, count in enumerate(self._counts):
                height = int((count / maximum) * max(1, plot_rect.height() - 2))
                x = int(plot_rect.left() + index * bar_width)
                y = int(plot_rect.bottom() - height)
                painter.drawRect(x, y, max(1, int(np.ceil(bar_width))), height)

        self._draw_threshold_line(painter, self._contrast_low, QColor("#ffffff"))
        self._draw_threshold_line(painter, self._contrast_high, QColor("#ffcc00"))

        painter.setPen(QColor("#c7c7c7"))
        painter.drawText(
            self.rect().adjusted(6, self.height() - 16, -6, 0),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"{self._edges[0]:.4g}",
        )
        painter.drawText(
            self.rect().adjusted(6, self.height() - 16, -6, 0),
            Qt.AlignRight | Qt.AlignVCenter,
            f"{self._edges[-1]:.4g}",
        )
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        threshold = self._threshold_near_position(self._event_x(event))
        if threshold is None:
            super().mousePressEvent(event)
            return
        self._dragging_threshold = threshold
        self._set_dragged_threshold(self._event_x(event))
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_threshold is not None:
            self._set_dragged_threshold(self._event_x(event))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging_threshold is None:
            super().mouseReleaseEvent(event)
            return
        self._set_dragged_threshold(self._event_x(event))
        self._dragging_threshold = None
        event.accept()

    def _draw_threshold_line(self, painter, value: float | None, color: QColor) -> None:
        x = self._value_to_x(value)
        if x is None:
            return
        plot_rect = self._plot_rect()
        painter.setPen(QPen(color, 2))
        painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())

    def _plot_rect(self):
        return self.rect().adjusted(6, 6, -6, -18)

    def _event_x(self, event) -> float:
        position = event.position() if hasattr(event, "position") else event.pos()
        return float(position.x())

    def _value_to_x(self, value: float | None) -> int | None:
        if value is None or self._edges.size < 2:
            return None
        histogram_low = float(self._edges[0])
        histogram_high = float(self._edges[-1])
        if histogram_high <= histogram_low:
            return None
        plot_rect = self._plot_rect()
        fraction = min(1.0, max(0.0, (float(value) - histogram_low) / (histogram_high - histogram_low)))
        return int(plot_rect.left() + fraction * plot_rect.width())

    def _x_to_value(self, x_position: float) -> float | None:
        if self._edges.size < 2:
            return None
        histogram_low = float(self._edges[0])
        histogram_high = float(self._edges[-1])
        if histogram_high <= histogram_low:
            return None
        plot_rect = self._plot_rect()
        fraction = (float(x_position) - plot_rect.left()) / max(1, plot_rect.width())
        fraction = min(1.0, max(0.0, fraction))
        return histogram_low + fraction * (histogram_high - histogram_low)

    def _threshold_near_position(self, x_position: float) -> str | None:
        low_x = self._value_to_x(self._contrast_low)
        high_x = self._value_to_x(self._contrast_high)
        candidates = []
        if low_x is not None:
            candidates.append(("low", abs(float(low_x) - x_position)))
        if high_x is not None:
            candidates.append(("high", abs(float(high_x) - x_position)))
        if not candidates:
            return None
        threshold, distance = min(candidates, key=lambda item: item[1])
        return threshold if distance <= 12.0 else None

    def _set_dragged_threshold(self, x_position: float) -> None:
        if self._dragging_threshold is None:
            return
        value = self._x_to_value(x_position)
        if value is None:
            return
        low = self._contrast_low
        high = self._contrast_high
        if low is None or high is None:
            return
        histogram_span = max(float(self._edges[-1] - self._edges[0]), 1.0)
        minimum_gap = histogram_span * 1e-9
        if self._dragging_threshold == "low":
            low = min(float(value), float(high) - minimum_gap)
        else:
            high = max(float(value), float(low) + minimum_gap)
        self._contrast_low = float(low)
        self._contrast_high = float(high)
        self.update()
        self.contrast_changed.emit(float(low), float(high))


class VolumeAnnotationWidget(QWidget):
    def __init__(self, viewer, initial_folder: Path | None = None) -> None:
        super().__init__()
        self.viewer = viewer
        self.records: Sequence[MrcSliceRecord] = []
        self._load_worker = None
        self._pending_session: NanotationSession | None = None
        self._volume_layer = None

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Folder containing successive MRC slices")
        self.browse_button = QPushButton("Browse")
        self.load_button = QPushButton("Load Volume")

        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit, stretch=1)
        folder_row.addWidget(self.browse_button)

        self.summary_label = QLabel("Choose a folder to load a volume.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.annotation_plot = Annotation3DPlot()

        self.coordinate_scale_spin = QDoubleSpinBox()
        self.coordinate_scale_spin.setRange(0.000001, 1_000_000.0)
        self.coordinate_scale_spin.setDecimals(6)
        self.coordinate_scale_spin.setValue(1.0)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("XY-Coordinate scale ouput"))
        scale_row.addWidget(self.coordinate_scale_spin, stretch=1)

        self.zoom_label = QLabel("Zoom: 1.00x")
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_reset_button = QPushButton("Zoom 1:1")
        self.zoom_in_button = QPushButton("Zoom In")

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.zoom_label, stretch=1)
        zoom_row.addWidget(self.zoom_out_button)
        zoom_row.addWidget(self.zoom_reset_button)
        zoom_row.addWidget(self.zoom_in_button)

        self.output_image_size_spin = QSpinBox()
        self.output_image_size_spin.setRange(1, 1_000_000)
        self.output_image_size_spin.setValue(DEFAULT_OUTPUT_IMAGE_SIZE)

        output_image_size_row = QHBoxLayout()
        output_image_size_row.addWidget(QLabel("Output Image Size"))
        output_image_size_row.addWidget(self.output_image_size_spin, stretch=1)

        self.histogram_widget = HistogramWidget()
        self.normalize_contrast_button = QPushButton("Normalize Contrast")
        self.normalize_contrast_button.setEnabled(False)

        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self.normalize_contrast_button)

        self.clear_points_button = QPushButton("Clear Points")
        self.export_button = QPushButton("Export Coordinates…")
        self.save_session_button = QPushButton("Save Session…")
        self.load_session_button = QPushButton("Load Session…")
        self.clear_points_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.save_session_button.setEnabled(False)

        session_row = QHBoxLayout()
        session_row.addWidget(self.save_session_button)
        session_row.addWidget(self.load_session_button)

        instructions = QLabel(
            "Scroll with napari's slice control, then click locations "
            "in any slice. Coordinates can be exported whenever needed."
        )
        instructions.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addLayout(folder_row)
        layout.addWidget(self.load_button)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.annotation_plot)
        layout.addWidget(self.clear_points_button)
        layout.addLayout(zoom_row)
        layout.addWidget(self.histogram_widget)
        layout.addLayout(contrast_row)
        layout.addLayout(output_image_size_row)
        layout.addLayout(scale_row)
        layout.addWidget(instructions)
        layout.addWidget(self.export_button)
        layout.addLayout(session_row)
        layout.addStretch(1)
        self.setLayout(layout)

        self.browse_button.clicked.connect(self._browse)
        self.load_button.clicked.connect(self.load_volume)
        self.zoom_out_button.clicked.connect(self._zoom_out)
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        self.zoom_in_button.clicked.connect(self._zoom_in)
        self.normalize_contrast_button.clicked.connect(self._normalize_contrast_to_current_frame)
        self.histogram_widget.contrast_changed.connect(self._set_contrast_limits)
        self.clear_points_button.clicked.connect(self._clear_points)
        self.export_button.clicked.connect(self._export_coordinates)
        self.save_session_button.clicked.connect(self._save_session)
        self.load_session_button.clicked.connect(self._load_session)
        self.annotation_plot.slice_requested.connect(self._go_to_slice)
        self.viewer.camera.events.zoom.connect(self._update_zoom_label)
        self.viewer.layers.selection.events.active.connect(self._hide_napari_layer_controls)
        self.viewer.dims.events.current_step.connect(self._update_plot_slice_index)
        self._apply_eman2_image_orientation()
        self._set_zoom(1.0)

        if initial_folder is not None:
            self.folder_edit.setText(str(initial_folder))
            self.load_volume()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select MRC slice folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            self.load_volume()

    def load_volume(self) -> None:
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            self._show_error("Choose a folder first.")
            return

        from napari.qt.threading import thread_worker

        self.load_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self.summary_label.setText("Indexing MRC slices…")
        worker = thread_worker(_prepare_volume, start_thread=False, ignore_errors=True)(
            Path(folder_text).expanduser()
        )
        worker.returned.connect(self._finish_volume_load)
        worker.errored.connect(self._volume_load_failed)
        worker.finished.connect(self._volume_load_finished)
        self._load_worker = worker
        worker.start()

    def _finish_volume_load(self, result) -> None:
        records, volume = result

        self._remove_layer(VOLUME_LAYER_NAME)
        self._remove_layer(POINTS_LAYER_NAME)
        self._remove_layer(INTERSECTION_LAYER_NAME)
        self.records = records
        self._volume_layer = None
        self._set_contrast_controls_enabled(False)
        scale = volume_scale(records)
        self.annotation_plot.set_volume_shape(len(records), records[0].shape)
        volume_layer = self.viewer.add_image(
            volume,
            name=VOLUME_LAYER_NAME,
            colormap="gray",
            scale=scale,
            metadata={
                "source_folder": str(records[0].path.parent),
                "slice_count": len(records),
            },
        )
        self._volume_layer = volume_layer
        self._set_default_interpolation(volume_layer)
        points = self.viewer.add_points(
            np.empty((0, 3), dtype=float),
            name=POINTS_LAYER_NAME,
            ndim=3,
            scale=scale,
            size=POINT_SIZE,
            face_color="transparent",
            border_color=POINT_BORDER_COLOR,
            border_width=0.2,
            opacity=POINT_OPACITY,
        )
        intersection = self.viewer.add_points(
            np.empty((0, 3), dtype=float),
            name=INTERSECTION_LAYER_NAME,
            ndim=3,
            scale=scale,
            size=INTERSECTION_SIZE,
            symbol="cross",
            face_color=INTERSECTION_COLOR,
            border_color=INTERSECTION_COLOR,
            border_width=0,
            opacity=INTERSECTION_OPACITY,
        )
        intersection.editable = False
        points.events.data.connect(self._update_point_count)
        self.viewer.dims.ndisplay = 2
        self._reset_slice_position()
        self.viewer.layers.selection.active = points
        points.mode = "add"

        for button in (
            self.clear_points_button,
            self.export_button,
            self.save_session_button,
        ):
            button.setEnabled(True)
        self._set_contrast_controls_enabled(True)
        self._normalize_contrast_to_current_frame()
        self._update_point_count()
        self._reset_zoom()
        self._hide_napari_layer_controls()
        QTimer.singleShot(0, self._hide_napari_layer_controls)
        if self._pending_session is not None:
            try:
                self._restore_session(self._pending_session, points)
            except ValueError as exc:
                self._show_error(str(exc))
            finally:
                self._pending_session = None

    def _volume_load_failed(self, error: Exception) -> None:
        self._pending_session = None
        self._show_error(str(error))

    def _volume_load_finished(self) -> None:
        self.load_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.load_session_button.setEnabled(True)
        self.save_session_button.setEnabled(bool(self.records))
        self._load_worker = None

    def _points_layer(self):
        if POINTS_LAYER_NAME not in self.viewer.layers:
            return None
        return self.viewer.layers[POINTS_LAYER_NAME]

    def _zoom_in(self) -> None:
        self._set_zoom(float(self.viewer.camera.zoom) * ZOOM_STEP)

    def _zoom_out(self) -> None:
        self._set_zoom(float(self.viewer.camera.zoom) / ZOOM_STEP)

    def _reset_zoom(self) -> None:
        if hasattr(self.viewer, "reset_view"):
            self.viewer.reset_view()
        elif hasattr(self.viewer, "fit_to_view"):
            self.viewer.fit_to_view()
        self._apply_eman2_image_orientation()
        self._set_zoom(1.0)

    def _set_zoom(self, zoom: float) -> None:
        self.viewer.camera.zoom = min(MAX_ZOOM, max(MIN_ZOOM, float(zoom)))
        self._update_zoom_label()

    def _apply_eman2_image_orientation(self) -> None:
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "orientation2d"):
            camera.orientation2d = EMAN2_IMAGE_ORIENTATION_2D

    def _update_zoom_label(self, event=None) -> None:
        del event
        self.zoom_label.setText(f"Zoom: {float(self.viewer.camera.zoom):.2f}x")

    def _reset_slice_position(self) -> None:
        if hasattr(self.viewer.dims, "set_current_step"):
            self.viewer.dims.set_current_step(0, 0)
            return
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if current_step:
            self.viewer.dims.current_step = (0, *current_step[1:])

    def _set_default_interpolation(self, layer) -> None:
        for attribute in ("interpolation2d", "interpolation3d"):
            if hasattr(layer, attribute):
                setattr(layer, attribute, DEFAULT_INTERPOLATION)

    def _set_contrast_controls_enabled(self, enabled: bool) -> None:
        self.normalize_contrast_button.setEnabled(enabled)

    def _current_slice_index(self) -> int | None:
        if not self.records:
            return None
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if not current_step:
            return 0
        return min(max(0, int(current_step[0])), len(self.records) - 1)

    def _current_slice_data(self) -> np.ndarray | None:
        if self._volume_layer is None:
            return None
        slice_index = self._current_slice_index()
        if slice_index is None:
            return None
        try:
            return np.asarray(self._volume_layer.data[slice_index])
        except (IndexError, OSError, ValueError):
            return None

    def _normalize_contrast_to_current_frame(self) -> None:
        standard_deviation = finite_intensity_standard_deviation(self._current_slice_data())
        if standard_deviation is None:
            self.histogram_widget.set_histogram(None)
            return
        limit = CONTRAST_STD_MULTIPLIER * standard_deviation
        if limit <= 0:
            limit = 1.0
        low, high = -limit, limit
        self._set_contrast_limits(low, high)

    def _set_contrast_limits(self, low: float, high: float) -> None:
        if self._volume_layer is None:
            return
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if high <= low:
            high = low + max(abs(low) * 1e-6, 1e-6)
        self._volume_layer.contrast_limits = (low, high)
        self._update_current_frame_histogram()

    def _contrast_limits(self) -> tuple[float | None, float | None]:
        if self._volume_layer is None:
            return None, None
        try:
            low, high = self._volume_layer.contrast_limits
        except (TypeError, ValueError):
            return None, None
        return float(low), float(high)

    def _update_current_frame_histogram(self) -> None:
        low, high = self._contrast_limits()
        self.histogram_widget.set_histogram(
            self._current_slice_data(),
            contrast_low=low,
            contrast_high=high,
        )

    def _hide_napari_layer_controls(self, event=None) -> None:
        del event
        qt_viewer = getattr(getattr(self.viewer, "window", None), "_qt_viewer", None)
        controls_container = getattr(qt_viewer, "controls", None)
        controls_widgets = []
        controls = getattr(controls_container, "currentWidget", lambda: None)()
        if controls is not None:
            controls_widgets.append(controls)
        controls_widgets.extend(getattr(controls_container, "widgets", {}).values())
        controls_to_hide = {
            "_opacity_blending_controls": (
                "blend_combobox",
                "blend_label",
            ),
            "_projection_mode_control": (
                "projection_combobox",
                "projection_combobox_label",
            ),
            "_text_visibility_control": (
                "text_disp_checkbox",
                "text_disp_label",
            ),
            "_out_slice_checkbox_control": (
                "out_of_slice_checkbox",
                "out_of_slice_checkbox_label",
            ),
        }
        for controls_widget in controls_widgets:
            self._hide_widget_controls(controls_widget, controls_to_hide)

    def _hide_widget_controls(
        self,
        controls,
        control_widgets: dict[str, tuple[str, ...]],
    ) -> None:
        if controls is None:
            return
        for control_name, widget_names in control_widgets.items():
            control = getattr(controls, control_name, None)
            if control is None:
                continue
            for widget_name in widget_names:
                widget = getattr(control, widget_name, None)
                if widget is not None:
                    widget.setVisible(False)

    def _update_plot_slice_index(self, event=None) -> None:
        del event
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if current_step:
            self.annotation_plot.set_slice_index(current_step[0])
            self._update_intersection_marker(current_step[0])
            self._update_current_frame_histogram()

    def _update_intersection_marker(self, slice_index: int | None = None) -> None:
        if INTERSECTION_LAYER_NAME not in self.viewer.layers:
            return
        points = self._points_layer()
        if points is None:
            return
        if slice_index is None:
            current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
            if not current_step:
                return
            slice_index = int(current_step[0])

        intersection = path_intersection_at_slice(points.data, slice_index)
        data = (
            np.empty((0, 3), dtype=float)
            if intersection is None
            else np.asarray(intersection, dtype=float).reshape(1, 3)
        )
        layer = self.viewer.layers[INTERSECTION_LAYER_NAME]
        if not np.array_equal(np.asarray(layer.data), data):
            layer.data = data
        layer.editable = False

    def _go_to_slice(self, slice_index: int) -> None:
        if not self.records:
            return
        index = min(max(0, int(slice_index)), len(self.records) - 1)
        if hasattr(self.viewer.dims, "set_current_step"):
            self.viewer.dims.set_current_step(0, index)
            return
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if current_step:
            self.viewer.dims.current_step = (index, *current_step[1:])

    def _save_session(self) -> None:
        points = self._points_layer()
        if points is None or not self.records:
            self._show_error("Load a volume before saving a session.")
            return

        initial_path = self.records[0].path.parent / "nanotation-session.nanotation.json"
        filename, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Nanotation session",
            str(initial_path),
            "Nanotation sessions (*.nanotation.json);;JSON files (*.json)",
        )
        if not filename:
            return

        path = Path(filename)
        if not path.name.lower().endswith(".json"):
            path = path.with_name(path.name + ".nanotation.json")
        current_step = tuple(getattr(self.viewer.dims, "current_step", (0,)))
        try:
            write_session_file(
                path,
                source_folder=self.records[0].path.parent,
                annotations=points.data,
                coordinate_scale=self.coordinate_scale_spin.value(),
                records=self.records,
                slice_index=int(current_step[0]),
                zoom=float(self.viewer.camera.zoom),
                camera_3d=self.annotation_plot.camera_state(),
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.summary_label.setText(f"Saved session to {path}")

    def _load_session(self) -> None:
        initial_folder = self.folder_edit.text().strip() or str(Path.home())
        filename, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Nanotation session",
            initial_folder,
            "Nanotation sessions (*.nanotation.json *.json);;All files (*)",
        )
        if not filename:
            return
        try:
            session = read_session_file(Path(filename))
        except ValueError as exc:
            self._show_error(str(exc))
            return
        if not session.source_folder.is_dir():
            self._show_error(f"Session image folder does not exist: {session.source_folder}")
            return

        self._pending_session = session
        self.folder_edit.setText(str(session.source_folder))
        self.load_volume()

    def _restore_session(self, session: NanotationSession, points) -> None:
        if session.image_count != len(self.records):
            raise ValueError(
                f"Session expects {session.image_count} images, but the folder contains "
                f"{len(self.records)}."
            )
        if (
            session.first_filename != self.records[0].name
            or session.last_filename != self.records[-1].name
        ):
            raise ValueError("The first or last image filename does not match the saved session.")
        slice_indices = np.rint(session.annotations[:, 0]).astype(int)
        if slice_indices.size and (
            np.any(slice_indices < 0) or np.any(slice_indices >= len(self.records))
        ):
            raise ValueError("The session contains annotations outside the loaded image volume.")
        if session.slice_index >= len(self.records):
            raise ValueError("The saved slice index is outside the loaded image volume.")

        self.coordinate_scale_spin.setValue(session.coordinate_scale)
        points.data = session.annotations.copy()
        self._go_to_slice(session.slice_index)
        self._set_zoom(session.zoom)
        self.annotation_plot.restore_camera_state(session.camera_3d)
        self.summary_label.setText(
            f"Restored {len(session.annotations)} points from session; "
            f"loaded {len(self.records)} slices"
        )

    def _clear_points(self) -> None:
        points = self._points_layer()
        if points is None or len(points.data) == 0:
            return
        answer = QMessageBox.question(
            self,
            "Clear annotations",
            f"Delete all {len(points.data)} points?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            points.data = np.empty((0, 3), dtype=float)

    def _export_coordinates(self) -> None:
        points = self._points_layer()
        if points is None:
            self._show_error("Load a volume before exporting coordinates.")
            return

        initial_path = Path(self.folder_edit.text()).expanduser() / "annotations.csv"
        filename, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export annotation coordinates",
            str(initial_path),
            "CSV files (*.csv)",
        )
        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        try:
            count = write_annotations_csv(
                path,
                points.data,
                self.records,
                coordinate_scale=self.coordinate_scale_spin.value(),
            )
        except Exception as exc:  # noqa: BLE001 - display export errors in the GUI.
            self._show_error(str(exc))
            return
        self.summary_label.setText(f"Exported {count} coordinate rows to {path}")

    def _update_point_count(self, event=None) -> None:
        del event
        points = self._points_layer()
        count = len(points.data) if points is not None else 0
        if self.records:
            coordinates = points.data if points is not None else np.empty((0, 3), dtype=float)
            shape = self.records[0].shape
            self.summary_label.setText(
                f"Loaded {len(self.records)} slices as volume "
                f"({len(self.records)} × {shape[0]} × {shape[1]}); {count} points"
            )
            self.annotation_plot.set_annotations(coordinates)
            self._update_intersection_marker()

    def _remove_layer(self, name: str) -> None:
        if name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[name])

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Nanotation", message)


def _prepare_volume(folder: Path) -> tuple[Sequence[MrcSliceRecord], LazyMrcVolume]:
    records = scan_mrc_folder(folder, recursive=False)
    return records, LazyMrcVolume(records)
