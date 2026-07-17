from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from qtpy.QtCore import QTimer, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPen
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

from .annotations import DEFAULT_PATH_SMOOTHNESS, SmoothedPath, write_annotations_csv
from .mrc_io import LazyMrcVolume, MrcFrameRecord, scan_mrc_folder, volume_scale
from .plot3d import Annotation3DPlot
from .sessions import NanotationSession, read_session_file, write_session_file


VOLUME_LAYER_NAME = "MRC volume"
POINTS_LAYER_NAME = "Path checkpoints"
INTERSECTION_LAYER_NAME = "Linear path"
POINT_BORDER_COLOR = "#0055ffff"
POINT_OPACITY = 0.6
POINT_SIZE = 32
INTERSECTION_COLOR = "#ffaa00ff"
INTERSECTION_OPACITY = 0.4
INTERSECTION_SIZE = 24
ZOOM_STEP = 1.25
MIN_ZOOM = 0.01
MAX_ZOOM = 100.0
DEFAULT_INTERPOLATION = "bicubic"
EMAN2_IMAGE_ORIENTATION_2D = ("up", "right")
HISTOGRAM_BINS = 96
INITIAL_DISPLAY_STD_MULTIPLIER = 5.0
INITIAL_CONTRAST_LOW_STD_MULTIPLIER = -3.0
INITIAL_CONTRAST_HIGH_STD_MULTIPLIER = 4.0


def finite_intensity_standard_deviation(values: np.ndarray | None) -> float | None:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    standard_deviation = float(finite.std())
    return standard_deviation if np.isfinite(standard_deviation) else None


def finite_symmetric_standard_deviation_limits(
    values: np.ndarray | None,
    multiplier: float,
) -> tuple[float, float] | None:
    return finite_standard_deviation_limits(values, -float(multiplier), float(multiplier))


def finite_standard_deviation_limits(
    values: np.ndarray | None,
    low_multiplier: float,
    high_multiplier: float,
) -> tuple[float, float] | None:
    standard_deviation = finite_intensity_standard_deviation(values)
    if standard_deviation is None:
        return None
    low_multiplier = float(low_multiplier)
    high_multiplier = float(high_multiplier)
    if (
        not np.isfinite(low_multiplier)
        or not np.isfinite(high_multiplier)
        or high_multiplier <= low_multiplier
    ):
        return None
    if standard_deviation <= 0:
        return -1.0, 1.0
    return low_multiplier * standard_deviation, high_multiplier * standard_deviation


class HistogramWidget(QWidget):
    """Compact histogram for the currently displayed image frame."""

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
        display_range: tuple[float, float] | None = None,
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
        if display_range is not None:
            display_low, display_high = (float(display_range[0]), float(display_range[1]))
            if (
                np.isfinite(display_low)
                and np.isfinite(display_high)
                and display_high > display_low
            ):
                data_min, data_max = display_low, display_high
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
        return self.rect().adjusted(6, 6, -6, -6)

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
        fraction = (float(value) - histogram_low) / (histogram_high - histogram_low)
        fraction = min(1.0, max(0.0, fraction))
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
        self.records: Sequence[MrcFrameRecord] = []
        self._load_worker = None
        self._pending_session: NanotationSession | None = None
        self._volume_layer = None
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
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_reset_button = QPushButton("Zoom 1:1")
        self.zoom_in_button = QPushButton("Zoom In")

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.zoom_label, stretch=1)
        zoom_row.addWidget(self.zoom_out_button)
        zoom_row.addWidget(self.zoom_reset_button)
        zoom_row.addWidget(self.zoom_in_button)

        self.histogram_widget = HistogramWidget()

        self.path_smoothness_spin = QDoubleSpinBox()
        self.path_smoothness_spin.setRange(0.0, 1.0)
        self.path_smoothness_spin.setDecimals(2)
        self.path_smoothness_spin.setSingleStep(0.05)
        self.path_smoothness_spin.setValue(DEFAULT_PATH_SMOOTHNESS)

        smoothness_row = QHBoxLayout()
        smoothness_row.addWidget(QLabel("Path Smoothness"))
        smoothness_row.addWidget(self.path_smoothness_spin, stretch=1)

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
            "Scroll with the frame control, then click locations "
            "in any frame. Coordinates can be exported whenever needed."
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
        layout.addLayout(smoothness_row)
        layout.addWidget(instructions)
        layout.addWidget(self.export_button)
        layout.addLayout(session_row)
        layout.addStretch(1)
        self.setLayout(layout)

        self.browse_button.clicked.connect(self._browse)
        self.load_button.clicked.connect(self.load_time_series)
        self.zoom_out_button.clicked.connect(self._zoom_out)
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        self.zoom_in_button.clicked.connect(self._zoom_in)
        self.histogram_widget.contrast_changed.connect(self._set_contrast_limits)
        self.path_smoothness_spin.valueChanged.connect(self._path_smoothness_changed)
        self.clear_points_button.clicked.connect(self._clear_points)
        self.export_button.clicked.connect(self._export_coordinates)
        self.save_session_button.clicked.connect(self._save_session)
        self.load_session_button.clicked.connect(self._load_session)
        self.annotation_plot.frame_requested.connect(self._go_to_frame)
        self.viewer.camera.events.zoom.connect(self._update_zoom_label)
        self.viewer.layers.selection.events.active.connect(self._hide_napari_layer_controls)
        self.viewer.dims.events.current_step.connect(self._update_plot_frame_index)
        self._apply_eman2_image_orientation()
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

        from napari.qt.threading import thread_worker

        self.load_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self.summary_label.setText("Indexing MRC frames…")
        worker = thread_worker(_prepare_time_series, start_thread=False, ignore_errors=True)(
            Path(folder_text).expanduser()
        )
        worker.returned.connect(self._finish_time_series_load)
        worker.errored.connect(self._time_series_load_failed)
        worker.finished.connect(self._time_series_load_finished)
        self._load_worker = worker
        worker.start()

    def _finish_time_series_load(self, result) -> None:
        records, volume = result

        self._remove_layer(VOLUME_LAYER_NAME)
        self._remove_layer(POINTS_LAYER_NAME)
        self._remove_layer(INTERSECTION_LAYER_NAME)
        self.records = records
        self._volume_layer = None
        self._contrast_display_range = None
        scale = volume_scale(records)
        self.annotation_plot.set_volume_shape(len(records), records[0].shape)
        volume_layer = self.viewer.add_image(
            volume,
            name=VOLUME_LAYER_NAME,
            colormap="gray",
            scale=scale,
            metadata={
                "source_folder": str(records[0].path.parent),
                "frame_count": len(records),
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

    def _time_series_load_failed(self, error: Exception) -> None:
        self._pending_session = None
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

    def _reset_frame_position(self) -> None:
        if hasattr(self.viewer.dims, "set_current_step"):
            self.viewer.dims.set_current_step(0, 0)
            return
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if current_step:
            self.viewer.dims.current_step = (0, *current_step[1:])

    def _configure_frame_controls(self) -> None:
        axis_labels = list(getattr(self.viewer.dims, "axis_labels", ()))
        if axis_labels:
            axis_labels[0] = "Frame"
            self.viewer.dims.axis_labels = tuple(axis_labels)
        QTimer.singleShot(0, self._update_frame_slider_labels)

    def _update_frame_slider_labels(self) -> None:
        frame_index = self._current_frame_index()
        if frame_index is None:
            return
        qt_viewer = getattr(getattr(self.viewer, "window", None), "_qt_viewer", None)
        qt_dims = getattr(qt_viewer, "dims", None)
        slider_widgets = getattr(qt_dims, "slider_widgets", ())
        if not slider_widgets:
            return
        frame_slider = slider_widgets[0]
        current_label = getattr(frame_slider, "curslice_label", None)
        total_label = getattr(frame_slider, "totslice_label", None)
        if current_label is None or total_label is None:
            return
        current_label.setReadOnly(True)
        current_label.setText(str(frame_index + 1))
        total_label.setText(str(len(self.records)))
        width = current_label.fontMetrics().horizontalAdvance("8" * len(str(len(self.records)))) + 6
        current_label.setFixedWidth(width)
        total_label.setFixedWidth(width)

    def _set_default_interpolation(self, layer) -> None:
        for attribute in ("interpolation2d", "interpolation3d"):
            if hasattr(layer, attribute):
                setattr(layer, attribute, DEFAULT_INTERPOLATION)

    def _current_frame_index(self) -> int | None:
        if not self.records:
            return None
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        if not current_step:
            return 0
        return min(max(0, int(current_step[0])), len(self.records) - 1)

    def _current_frame_data(self) -> np.ndarray | None:
        if self._volume_layer is None:
            return None
        frame_index = self._current_frame_index()
        if frame_index is None:
            return None
        try:
            return np.asarray(self._volume_layer.data[frame_index])
        except (IndexError, OSError, ValueError):
            return None

    def _set_initial_contrast_from_current_frame(self) -> None:
        data = self._current_frame_data()
        display_range = finite_symmetric_standard_deviation_limits(
            data,
            INITIAL_DISPLAY_STD_MULTIPLIER,
        )
        contrast_limits = finite_standard_deviation_limits(
            data,
            INITIAL_CONTRAST_LOW_STD_MULTIPLIER,
            INITIAL_CONTRAST_HIGH_STD_MULTIPLIER,
        )
        if display_range is None or contrast_limits is None:
            self.histogram_widget.set_histogram(None)
            return
        self._set_contrast_display_range(*display_range)
        self._set_contrast_limits(*contrast_limits)

    def _set_contrast_display_range(self, low: float, high: float) -> None:
        if self._volume_layer is None:
            return
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return
        self._contrast_display_range = (low, high)
        if hasattr(self._volume_layer, "contrast_limits_range"):
            self._volume_layer.contrast_limits_range = (low, high)

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

    def _contrast_display_limits_range(self) -> tuple[float, float] | None:
        if self._volume_layer is not None and hasattr(self._volume_layer, "contrast_limits_range"):
            try:
                low, high = self._volume_layer.contrast_limits_range
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

    def _update_plot_frame_index(self, event=None) -> None:
        del event
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
        self.load_time_series()

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
        frame_indices = np.rint(session.annotations[:, 0]).astype(int)
        if frame_indices.size and (
            np.any(frame_indices < 0) or np.any(frame_indices >= len(self.records))
        ):
            raise ValueError("The session contains annotations outside the loaded image volume.")
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
                self.path_smoothness_spin.value(),
            )
        except (OSError, ValueError, csv.Error) as exc:
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

    def _path_smoothness_changed(self, value: float) -> None:
        del value
        self._update_point_count()

    def _remove_layer(self, name: str) -> None:
        if name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[name])

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Nanotation", message)


def _prepare_time_series(folder: Path) -> tuple[Sequence[MrcFrameRecord], LazyMrcVolume]:
    records = scan_mrc_folder(folder, recursive=False)
    return records, LazyMrcVolume(records)
