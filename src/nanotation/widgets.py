from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .annotations import DEFAULT_PATH_SMOOTHNESS, SmoothedPath, write_annotation_entries
from .histogram import (
    HistogramWidget,
    finite_intensity_standard_deviation,
    standard_deviation_limits,
)
from .mrc_io import LazyMrcTimeSeries, MrcFrameRecord, scan_mrc_folder, time_series_scale
from .napari_ui import (
    apply_eman2_image_orientation,
    hide_layer_controls,
    set_default_image_interpolation,
    set_frame_axis_label,
    update_frame_slider_labels,
)
from .plot3d import Annotation3DPlot
from .sessions import NanotationSession, read_session_file, write_session_file


IMAGE_LAYER_NAME = "MRC time-series"
POINTS_LAYER_NAME = "Path checkpoints"
INTERSECTION_LAYER_NAME = "Smooth path"
POINT_BORDER_COLOR = "#0055ffff"
POINT_OPACITY = 0.6
IMAGE_CHECKPOINT_SIZE = 32
INTERSECTION_COLOR = "#ffaa00ff"
INTERSECTION_OPACITY = 0.4
INTERSECTION_SIZE = 24
MIN_ZOOM = 0.01
MAX_ZOOM = 100.0
INITIAL_DISPLAY_STD_MULTIPLIER = 5.0
INITIAL_CONTRAST_LOW_STD_MULTIPLIER = -3.0
INITIAL_CONTRAST_HIGH_STD_MULTIPLIER = 4.0


@dataclass(frozen=True, slots=True)
class _RefreshAnnotations:
    source_folder: Path
    checkpoints: tuple[tuple[str, float, float], ...]


def _capture_refresh_annotations(
    coordinates: np.ndarray,
    records: Sequence[MrcFrameRecord],
) -> _RefreshAnnotations | None:
    data = np.asarray(coordinates, dtype=float)
    if not records or data.ndim != 2 or data.shape[1] != 3:
        return None

    checkpoints = []
    for frame_value, y_value, x_value in data:
        if not np.all(np.isfinite((frame_value, y_value, x_value))):
            continue
        frame_index = int(round(frame_value))
        if 0 <= frame_index < len(records):
            checkpoints.append(
                (records[frame_index].name, float(y_value), float(x_value))
            )
    if not checkpoints:
        return None
    return _RefreshAnnotations(records[0].path.parent.resolve(), tuple(checkpoints))


def _remap_refresh_annotations(
    annotations: _RefreshAnnotations,
    records: Sequence[MrcFrameRecord],
) -> tuple[np.ndarray, int]:
    if not records or records[0].path.parent.resolve() != annotations.source_folder:
        return np.empty((0, 3), dtype=float), len(annotations.checkpoints)

    wanted_names = {checkpoint[0] for checkpoint in annotations.checkpoints}
    frame_indices = {
        record.name: frame_index
        for frame_index, record in enumerate(records)
        if record.name in wanted_names
    }
    remapped = [
        (frame_indices[filename], y_value, x_value)
        for filename, y_value, x_value in annotations.checkpoints
        if filename in frame_indices
    ]
    coordinates = (
        np.asarray(remapped, dtype=float)
        if remapped
        else np.empty((0, 3), dtype=float)
    )
    return coordinates, len(annotations.checkpoints) - len(remapped)


class NanotationWidget(QWidget):
    def __init__(self, viewer, initial_folder: Path | None = None) -> None:
        super().__init__()
        self.viewer = viewer
        self.records: Sequence[MrcFrameRecord] = []
        self._load_worker = None
        self._pending_session: NanotationSession | None = None
        self._pending_refresh_annotations: _RefreshAnnotations | None = None
        self._image_layer = None
        self._contrast_display_range: tuple[float, float] | None = None
        self._smoothed_path = SmoothedPath(np.empty((0, 3), dtype=float))

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Folder containing successive MRC frames")
        self.browse_button = QPushButton("Browse")
        self.load_button = QPushButton("Load/Refresh Time-series")

        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit, stretch=1)
        folder_row.addWidget(self.browse_button)

        self.summary_label = QLabel("Choose a folder to load a time-series.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.annotation_plot = Annotation3DPlot()

        self.zoom_label = QLabel("Zoom: 1.00x")
        self.zoom_reset_button = QPushButton("Zoom 1:1")

        self.histogram_widget = HistogramWidget()

        self.path_smoothness_spin = QDoubleSpinBox()
        self.path_smoothness_spin.setRange(0.0, 1.0)
        self.path_smoothness_spin.setDecimals(2)
        self.path_smoothness_spin.setSingleStep(0.05)
        self.path_smoothness_spin.setValue(DEFAULT_PATH_SMOOTHNESS)

        smoothness_row = QHBoxLayout()
        smoothness_row.addWidget(QLabel("Path Smoothness"))
        smoothness_row.addWidget(self.path_smoothness_spin, stretch=1)
        smoothness_row.addWidget(self.zoom_label)
        smoothness_row.addWidget(self.zoom_reset_button)

        self.clear_points_button = QPushButton("Clear Points")
        self.export_button = QPushButton("Export track…")
        self.save_session_button = QPushButton("Save Session…")
        self.load_session_button = QPushButton("Load Session…")
        self.clear_points_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.save_session_button.setEnabled(False)

        session_row = QHBoxLayout()
        session_row.addWidget(self.save_session_button)
        session_row.addWidget(self.load_session_button)

        layout = QVBoxLayout()
        layout.addLayout(folder_row)
        layout.addWidget(self.load_button)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.annotation_plot)
        layout.addWidget(self.clear_points_button)
        layout.addWidget(self.histogram_widget)
        layout.addLayout(smoothness_row)
        layout.addWidget(self.export_button)
        layout.addLayout(session_row)
        layout.addStretch(1)
        self.setLayout(layout)

        self.browse_button.clicked.connect(self._browse)
        self.load_button.clicked.connect(self.load_time_series)
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        self.histogram_widget.contrast_changed.connect(self._set_contrast_limits)
        self.path_smoothness_spin.valueChanged.connect(self._update_point_count)
        self.clear_points_button.clicked.connect(self._clear_points)
        self.export_button.clicked.connect(self._export_coordinates)
        self.save_session_button.clicked.connect(self._save_session)
        self.load_session_button.clicked.connect(self._load_session)
        self.annotation_plot.frame_requested.connect(self._go_to_frame)
        self.viewer.camera.events.zoom.connect(self._update_zoom_label)
        self.viewer.layers.selection.events.active.connect(self._hide_napari_layer_controls)
        self.viewer.dims.events.current_step.connect(self._update_plot_frame_index)
        apply_eman2_image_orientation(self.viewer)
        self._set_zoom(1.0)

        if initial_folder is not None:
            self.folder_edit.setText(str(initial_folder))
            self.load_time_series()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select MRC frame folder",
            self.folder_edit.text(),
        )
        if folder:
            self.folder_edit.setText(folder)
            self.load_time_series()

    def load_time_series(self) -> None:
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            self._show_error("Choose a folder first.")
            return

        self._pending_refresh_annotations = None
        points = self._points_layer()
        target_folder = Path(folder_text).expanduser().resolve()
        current_folder = self.records[0].path.parent.resolve() if self.records else None
        if (
            self._pending_session is None
            and points is not None
            and target_folder == current_folder
        ):
            self._pending_refresh_annotations = _capture_refresh_annotations(
                points.data,
                self.records,
            )

        from napari.qt.threading import thread_worker

        self.load_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self.summary_label.setText("Indexing MRC frames…")
        worker = thread_worker(_prepare_time_series, start_thread=False, ignore_errors=True)(
            target_folder
        )
        worker.returned.connect(self._finish_time_series_load)
        worker.errored.connect(self._time_series_load_failed)
        worker.finished.connect(self._time_series_load_finished)
        self._load_worker = worker
        worker.start()

    def _finish_time_series_load(self, result) -> None:
        records, time_series = result

        self._remove_layer(IMAGE_LAYER_NAME)
        self._remove_layer(POINTS_LAYER_NAME)
        self._remove_layer(INTERSECTION_LAYER_NAME)
        self.records = records
        self._image_layer = None
        self._contrast_display_range = None
        scale = time_series_scale(records)
        self.annotation_plot.set_time_series_shape(len(records), records[0].shape)
        image_layer = self.viewer.add_image(
            time_series,
            name=IMAGE_LAYER_NAME,
            colormap="gray",
            scale=scale,
            metadata={
                "source_folder": str(records[0].path.parent),
                "frame_count": len(records),
            },
        )
        self._image_layer = image_layer
        set_default_image_interpolation(image_layer)
        points = self.viewer.add_points(
            np.empty((0, 3), dtype=float),
            name=POINTS_LAYER_NAME,
            ndim=3,
            scale=scale,
            size=IMAGE_CHECKPOINT_SIZE,
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
        self._reset_frame_position()
        self._configure_frame_controls()
        self.viewer.layers.selection.active = points
        points.mode = "add"

        for button in (
            self.clear_points_button,
            self.export_button,
            self.save_session_button,
        ):
            button.setEnabled(True)
        self._set_initial_contrast_from_current_frame()
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
        elif self._pending_refresh_annotations is not None:
            remapped, removed_count = _remap_refresh_annotations(
                self._pending_refresh_annotations,
                records,
            )
            points.data = remapped
            self._update_point_count()
            self.summary_label.setText(
                f"Refreshed {len(records)} frames; preserved {len(remapped)} checkpoints"
                + (
                    f" and removed {removed_count} from deleted files"
                    if removed_count
                    else ""
                )
            )
        self._pending_refresh_annotations = None

    def _time_series_load_failed(self, error: Exception) -> None:
        self._pending_session = None
        self._pending_refresh_annotations = None
        self._show_error(str(error))

    def _time_series_load_finished(self) -> None:
        self.load_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.load_session_button.setEnabled(True)
        self.save_session_button.setEnabled(bool(self.records))
        self._load_worker = None

    def _points_layer(self):
        if POINTS_LAYER_NAME not in self.viewer.layers:
            return None
        return self.viewer.layers[POINTS_LAYER_NAME]

    def _reset_zoom(self) -> None:
        if hasattr(self.viewer, "reset_view"):
            self.viewer.reset_view()
        elif hasattr(self.viewer, "fit_to_view"):
            self.viewer.fit_to_view()
        apply_eman2_image_orientation(self.viewer)
        self._set_zoom(1.0)

    def _set_zoom(self, zoom: float) -> None:
        self.viewer.camera.zoom = min(MAX_ZOOM, max(MIN_ZOOM, float(zoom)))
        self._update_zoom_label()

    def _update_zoom_label(self, _event=None) -> None:
        self.zoom_label.setText(f"Zoom: {float(self.viewer.camera.zoom):.2f}x")

    def _reset_frame_position(self) -> None:
        if hasattr(self.viewer.dims, "set_current_step"):
            self.viewer.dims.set_current_step(0, 0)
            return
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if current_step:
            self.viewer.dims.current_step = (0, *current_step[1:])

    def _configure_frame_controls(self) -> None:
        set_frame_axis_label(self.viewer)
        QTimer.singleShot(0, self._update_frame_slider_labels)

    def _update_frame_slider_labels(self) -> None:
        frame_index = self._current_frame_index()
        if frame_index is None:
            return
        update_frame_slider_labels(self.viewer, frame_index, len(self.records))

    def _current_frame_index(self) -> int | None:
        if not self.records:
            return None
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if not current_step:
            return 0
        return min(max(0, int(current_step[0])), len(self.records) - 1)

    def _current_frame_data(self) -> np.ndarray | None:
        if self._image_layer is None:
            return None
        frame_index = self._current_frame_index()
        if frame_index is None:
            return None
        try:
            return np.asarray(self._image_layer.data[frame_index])
        except (IndexError, OSError, ValueError):
            return None

    def _set_initial_contrast_from_current_frame(self) -> None:
        data = self._current_frame_data()
        standard_deviation = finite_intensity_standard_deviation(data)
        if standard_deviation is None:
            self.histogram_widget.set_histogram(None)
            return
        display_range = standard_deviation_limits(
            standard_deviation,
            -INITIAL_DISPLAY_STD_MULTIPLIER,
            INITIAL_DISPLAY_STD_MULTIPLIER,
        )
        contrast_limits = standard_deviation_limits(
            standard_deviation,
            INITIAL_CONTRAST_LOW_STD_MULTIPLIER,
            INITIAL_CONTRAST_HIGH_STD_MULTIPLIER,
        )
        if display_range is None or contrast_limits is None:
            self.histogram_widget.set_histogram(None)
            return
        self._set_contrast_display_range(*display_range)
        self._set_contrast_limits(*contrast_limits)

    def _set_contrast_display_range(self, low: float, high: float) -> None:
        if self._image_layer is None:
            return
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return
        self._contrast_display_range = (low, high)
        if hasattr(self._image_layer, "contrast_limits_range"):
            self._image_layer.contrast_limits_range = (low, high)

    def _set_contrast_limits(self, low: float, high: float) -> None:
        if self._image_layer is None:
            return
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if high <= low:
            high = low + max(abs(low) * 1e-6, 1e-6)
        self._image_layer.contrast_limits = (low, high)
        self._update_current_frame_histogram()

    def _contrast_limits(self) -> tuple[float | None, float | None]:
        if self._image_layer is None:
            return None, None
        try:
            low, high = self._image_layer.contrast_limits
        except (TypeError, ValueError):
            return None, None
        return float(low), float(high)

    def _contrast_display_limits_range(self) -> tuple[float, float] | None:
        if self._image_layer is not None and hasattr(self._image_layer, "contrast_limits_range"):
            try:
                low, high = self._image_layer.contrast_limits_range
                low = float(low)
                high = float(high)
                if np.isfinite(low) and np.isfinite(high) and high > low:
                    return low, high
            except (TypeError, ValueError):
                pass
        return self._contrast_display_range

    def _update_current_frame_histogram(self) -> None:
        low, high = self._contrast_limits()
        self.histogram_widget.set_histogram(
            self._current_frame_data(),
            contrast_low=low,
            contrast_high=high,
            display_range=self._contrast_display_limits_range(),
        )

    def _hide_napari_layer_controls(self, _event=None) -> None:
        hide_layer_controls(self.viewer)

    def _update_plot_frame_index(self, _event=None) -> None:
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if current_step:
            self.annotation_plot.set_frame_index(current_step[0])
            self._update_intersection_marker(current_step[0])
            self._update_current_frame_histogram()
            QTimer.singleShot(0, self._update_frame_slider_labels)

    def _update_intersection_marker(self, frame_index: int | None = None) -> None:
        if INTERSECTION_LAYER_NAME not in self.viewer.layers:
            return
        points = self._points_layer()
        if points is None:
            return
        if frame_index is None:
            current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
            if not current_step:
                return
            frame_index = int(current_step[0])

        intersection = self._smoothed_path.at_frame(frame_index)
        data = (
            np.empty((0, 3), dtype=float)
            if intersection is None
            else np.asarray(intersection, dtype=float).reshape(1, 3)
        )
        layer = self.viewer.layers[INTERSECTION_LAYER_NAME]
        if not np.array_equal(np.asarray(layer.data), data):
            layer.data = data
        layer.editable = False

    def _go_to_frame(self, frame_index: int) -> None:
        if not self.records:
            return
        index = min(max(0, int(frame_index)), len(self.records) - 1)
        if hasattr(self.viewer.dims, "set_current_step"):
            self.viewer.dims.set_current_step(0, index)
            return
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if current_step:
            self.viewer.dims.current_step = (index, *current_step[1:])

    def _save_session(self) -> None:
        points = self._points_layer()
        if points is None or not self.records:
            self._show_error("Load a time-series before saving a session.")
            return

        initial_path = self.records[0].path.parent / "nanotation-session.nanotation.json"
        filename = QFileDialog.getSaveFileName(
            self,
            "Save Nanotation session",
            str(initial_path),
            "Nanotation sessions (*.nanotation.json);;JSON files (*.json)",
        )[0]
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
                records=self.records,
                frame_number=int(current_step[0]) + 1,
                zoom=float(self.viewer.camera.zoom),
                camera_3d=self.annotation_plot.camera_state(),
                path_smoothness=self.path_smoothness_spin.value(),
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.summary_label.setText(f"Saved session to {path}")

    def _load_session(self) -> None:
        initial_folder = self.folder_edit.text().strip() or str(Path.home())
        filename = QFileDialog.getOpenFileName(
            self,
            "Load Nanotation session",
            initial_folder,
            "Nanotation sessions (*.nanotation.json *.json);;All files (*)",
        )[0]
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
        self.load_time_series()

    def _restore_session(self, session: NanotationSession, points) -> None:
        if session.frame_count != len(self.records):
            raise ValueError(
                f"Session expects {session.frame_count} frames, but the folder contains "
                f"{len(self.records)}."
            )
        if (
            session.first_filename != self.records[0].name
            or session.last_filename != self.records[-1].name
        ):
            raise ValueError("The first or last image filename does not match the saved session.")
        frame_indices = np.rint(session.annotations[:, 0]).astype(int)
        if frame_indices.size and (
            np.any(frame_indices < 0) or np.any(frame_indices >= len(self.records))
        ):
            raise ValueError("The session contains annotations outside the loaded time-series.")
        if session.frame_number > len(self.records):
            raise ValueError("The saved frame number is outside the loaded time-series.")

        self.path_smoothness_spin.blockSignals(True)
        self.path_smoothness_spin.setValue(session.path_smoothness)
        self.path_smoothness_spin.blockSignals(False)
        points.data = session.annotations.copy()
        self._go_to_frame(session.frame_number - 1)
        self._set_zoom(session.zoom)
        self.annotation_plot.restore_camera_state(session.camera_3d)
        self.summary_label.setText(
            f"Restored {len(session.annotations)} points from session; "
            f"loaded {len(self.records)} frames"
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
            self._show_error("Load a time-series before exporting coordinates.")
            return

        initial_path = Path(self.folder_edit.text()).expanduser() / "annotations.txt"
        filename = QFileDialog.getSaveFileName(
            self,
            "Export track",
            str(initial_path),
            "Text files (*.txt)",
        )[0]
        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")
        try:
            count = write_annotation_entries(
                path,
                points.data,
                self.records,
                self.path_smoothness_spin.value(),
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.summary_label.setText(f"Exported {count} track entries to {path}")

    def _update_point_count(self, _event=None) -> None:
        points = self._points_layer()
        count = len(points.data) if points is not None else 0
        if self.records:
            coordinates = points.data if points is not None else np.empty((0, 3), dtype=float)
            shape = self.records[0].shape
            self._smoothed_path = SmoothedPath(
                coordinates,
                self.path_smoothness_spin.value(),
                shape,
            )
            self.summary_label.setText(
                f"Loaded {len(self.records)} frames as a time-series "
                f"({len(self.records)} × {shape[0]} × {shape[1]}); {count} points"
            )
            self.annotation_plot.set_annotations(coordinates, self._smoothed_path.sample())
            self._update_intersection_marker()

    def _remove_layer(self, name: str) -> None:
        if name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[name])

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Nanotation", message)


def _prepare_time_series(
    folder: Path,
) -> tuple[Sequence[MrcFrameRecord], LazyMrcTimeSeries]:
    records = scan_mrc_folder(folder)
    return records, LazyMrcTimeSeries(records)
