from __future__ import annotations

import argparse
from pathlib import Path

import napari
from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QAbstractButton

from .widgets import VolumeAnnotationWidget


INITIAL_DOCK_WIDTH = 440


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View successive MRC slices and annotate points in Nanotation.")
    parser.add_argument("folder", nargs="?", type=Path, help="Folder containing successive MRC slices.")
    return parser


def run(folder: Path | None = None) -> None:
    viewer = napari.Viewer(title="Nanotation")
    viewer.camera.zoom = 1.0
    _hide_ndisplay_button(viewer)
    widget = VolumeAnnotationWidget(viewer, initial_folder=folder)
    dock_widget = viewer.window.add_dock_widget(widget, name="Nanotation", area="right")

    def resize_initial_dock() -> None:
        main_window = dock_widget.parentWidget()
        if main_window is not None and hasattr(main_window, "resizeDocks"):
            main_window.resizeDocks([dock_widget], [INITIAL_DOCK_WIDTH], Qt.Horizontal)

    QTimer.singleShot(0, resize_initial_dock)
    QTimer.singleShot(0, lambda: _hide_ndisplay_button(viewer))
    napari.run()


def _hide_ndisplay_button(viewer) -> None:
    qt_viewer = getattr(getattr(viewer, "window", None), "_qt_viewer", None)
    viewer_buttons = getattr(qt_viewer, "viewerButtons", None)
    ndisplay_button = getattr(viewer_buttons, "ndisplayButton", None)
    if ndisplay_button is not None:
        ndisplay_button.setVisible(False)

    qt_window = getattr(getattr(viewer, "window", None), "_qt_window", None)
    if qt_window is None:
        return
    for button in qt_window.findChildren(QAbstractButton):
        if "Toggle 2D/3D view" in button.toolTip():
            button.setVisible(False)


def main() -> None:
    args = build_parser().parse_args()
    run(args.folder)


if __name__ == "__main__":
    main()
