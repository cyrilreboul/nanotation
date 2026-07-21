from __future__ import annotations

import argparse
from pathlib import Path

import napari
from qtpy.QtCore import QTimer, Qt

from .napari_ui import hide_viewer_mode_buttons
from .widgets import NanotationWidget


INITIAL_DOCK_WIDTH = 440


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View successive MRC frames and annotate points in Nanotation."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Folder containing successive MRC frames.",
    )
    return parser


def run(folder: Path | None = None) -> None:
    viewer = napari.Viewer(title="Nanotation")
    viewer.camera.zoom = 1.0
    hide_viewer_mode_buttons(viewer)
    widget = NanotationWidget(viewer, initial_folder=folder)
    dock_widget = viewer.window.add_dock_widget(widget, name="Nanotation", area="right")

    def resize_initial_dock() -> None:
        main_window = dock_widget.parentWidget()
        if main_window is not None and hasattr(main_window, "resizeDocks"):
            main_window.resizeDocks([dock_widget], [INITIAL_DOCK_WIDTH], Qt.Horizontal)

    QTimer.singleShot(0, resize_initial_dock)
    QTimer.singleShot(0, lambda: hide_viewer_mode_buttons(viewer))
    napari.run()


def main() -> None:
    args = build_parser().parse_args()
    run(args.folder)


if __name__ == "__main__":
    main()
