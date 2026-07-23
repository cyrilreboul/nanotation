from __future__ import annotations

import numpy as np
from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from .annotations import annotation_xyz_coordinates, dashed_neighbor_segments


ANNOTATION_PLOT_SIZE = (400, 500)
CHECKPOINT_MARKER_SIZE = 14
CHECKPOINT_MARKER_FACE_COLOR = (0.0, 0.333, 1.0, 0.75)
CHECKPOINT_MARKER_EDGE_COLOR = (0.5, 0.8, 1.0, 1.0)


def bounding_box_segments(frame_count: int, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    x_extent = max(1, width - 1)
    y_extent = max(1, height - 1)
    z_extent = max(1, frame_count - 1)
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


class Annotation3DPlot(QWidget):
    """Read-only, rotatable 3D view of annotation coordinates and path."""

    frame_requested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        from vispy import scene

        self._scene = scene
        self._time_series_shape: tuple[int, int, int] | None = None
        self._xyz = np.empty((0, 3), dtype=np.float32)
        self._mouse_press_position: tuple[float, float] | None = None
        self._selection_handled_on_press = False
        self._canvas = scene.SceneCanvas(
            keys=None,
            show=False,
            size=ANNOTATION_PLOT_SIZE,
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
        self._frame_outline = scene.visuals.Line(
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
        self._set_marker_data(self._xyz)
        self._axis = scene.visuals.XYZAxis(parent=self._view.scene)

        instructions = QLabel("3D annotations — click a point to jump")
        instructions.setWordWrap(True)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(instructions)
        layout.addWidget(self._canvas.native, stretch=1)
        self.setLayout(layout)
        self.setMinimumHeight(180)
        self._canvas.native.setMinimumHeight(150)
        self._frame_label = QLabel("Frame: —", self._canvas.native)
        self._frame_label.setStyleSheet(
            "color: white; background-color: rgba(0, 0, 0, 150); padding: 3px 6px;"
        )
        self._frame_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._frame_label.move(8, 8)
        self._frame_label.adjustSize()
        self._frame_label.raise_()
        self.setToolTip("Read-only view; points and connecting line are ordered by frame.")
        self._canvas.native.installEventFilter(self)
        self._canvas.events.mouse_press.connect(self._on_canvas_mouse_press, position="first")
        self._markers.interactive = True
        self._markers.events.mouse_press.connect(self._on_marker_mouse_press)

    def set_time_series_shape(self, frame_count: int, image_shape: tuple[int, int]) -> None:
        height, width = image_shape
        time_series_shape = (int(frame_count), int(height), int(width))
        if time_series_shape == self._time_series_shape:
            return
        self._time_series_shape = time_series_shape

        x_extent = max(1, width - 1)
        y_extent = max(1, height - 1)
        z_extent = max(1, frame_count - 1)
        self._camera.set_range(
            x=(0, x_extent),
            y=(0, y_extent),
            z=(0, z_extent),
            margin=0.08,
        )
        self._bounding_box.set_data(
            pos=bounding_box_segments(frame_count, image_shape),
            connect="segments",
        )
        axis_size = max(x_extent, y_extent, z_extent) * 0.12
        self._axis.transform = self._scene.transforms.STTransform(
            scale=(axis_size, axis_size, axis_size)
        )
        self.set_frame_index(0)

    def set_frame_index(self, frame_index: int) -> None:
        if self._time_series_shape is None:
            return
        frame_count, height, width = self._time_series_shape
        index = min(max(0, int(frame_index)), max(0, frame_count - 1))
        self._frame_label.setText(f"Frame: {index + 1}")
        self._frame_label.adjustSize()

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
        self._frame_outline.set_data(pos=outline, connect="strip")
        self._canvas.update()

    def set_annotations(
        self,
        coordinates: np.ndarray,
        path_coordinates: np.ndarray | None = None,
    ) -> None:
        xyz = annotation_xyz_coordinates(coordinates).astype(np.float32, copy=False)
        self._xyz = xyz
        self._set_marker_data(xyz)
        if path_coordinates is None:
            path_coordinates = coordinates
        path_xyz = annotation_xyz_coordinates(path_coordinates).astype(np.float32, copy=False)
        self._line.set_data(
            pos=dashed_neighbor_segments(path_xyz, dash_count=1, dash_fraction=0.65),
            connect="segments",
        )
        self._canvas.update()

    def _set_marker_data(self, xyz: np.ndarray) -> None:
        self._markers.set_data(
            xyz,
            size=CHECKPOINT_MARKER_SIZE,
            face_color=CHECKPOINT_MARKER_FACE_COLOR,
            edge_color=CHECKPOINT_MARKER_EDGE_COLOR,
            edge_width=1,
        )

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
            if values["scale_factor"] <= 0:
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
        frame_index = int(np.rint(self._xyz[point_index, 2]))
        self.frame_requested.emit(frame_index)
        self._frame_label.setText(f"Frame: {frame_index + 1} — selected from 3D")
        self._frame_label.adjustSize()
        return True
