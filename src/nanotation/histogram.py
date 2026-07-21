from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QWidget


HISTOGRAM_BINS = 96


def finite_intensity_standard_deviation(values: np.ndarray | None) -> float | None:
    if values is None:
        return None
    data = np.asarray(values)
    if data.size == 0:
        return None
    finite_mask = np.isfinite(data)
    finite = data if bool(finite_mask.all()) else data[finite_mask]
    if finite.size == 0:
        return None
    standard_deviation = float(finite.std())
    return standard_deviation if np.isfinite(standard_deviation) else None


def standard_deviation_limits(
    standard_deviation: float,
    low_multiplier: float,
    high_multiplier: float,
) -> tuple[float, float] | None:
    standard_deviation = float(standard_deviation)
    low_multiplier = float(low_multiplier)
    high_multiplier = float(high_multiplier)
    if (
        not np.isfinite(standard_deviation)
        or not np.isfinite(low_multiplier)
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
            self._clear()
            return

        data = np.asarray(values)
        finite_mask = np.isfinite(data)
        if not bool(finite_mask.all()):
            data = data[finite_mask]
        if data.size == 0:
            self._clear()
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
                x_position = int(plot_rect.left() + index * bar_width)
                y_position = int(plot_rect.bottom() - height)
                painter.drawRect(
                    x_position,
                    y_position,
                    max(1, int(np.ceil(bar_width))),
                    height,
                )

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

    def _clear(self) -> None:
        self._counts = np.empty(0, dtype=float)
        self._edges = np.empty(0, dtype=float)
        self.update()

    def _draw_threshold_line(self, painter, value: float | None, color: QColor) -> None:
        x_position = self._value_to_x(value)
        if x_position is None:
            return
        plot_rect = self._plot_rect()
        painter.setPen(QPen(color, 2))
        painter.drawLine(x_position, plot_rect.top(), x_position, plot_rect.bottom())

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
