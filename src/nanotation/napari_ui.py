from __future__ import annotations

from qtpy.QtWidgets import QAbstractButton


DEFAULT_INTERPOLATION = "spline36"
EMAN2_IMAGE_ORIENTATION_2D = ("up", "right")

LAYER_CONTROLS_TO_HIDE = {
    "_opacity_blending_controls": ("blend_combobox", "blend_label"),
    "_projection_mode_control": ("projection_combobox", "projection_combobox_label"),
    "_text_visibility_control": ("text_disp_checkbox", "text_disp_label"),
    "_out_slice_checkbox_control": (
        "out_of_slice_checkbox",
        "out_of_slice_checkbox_label",
    ),
}


def apply_eman2_image_orientation(viewer) -> None:
    camera = getattr(viewer, "camera", None)
    if camera is not None and hasattr(camera, "orientation2d"):
        camera.orientation2d = EMAN2_IMAGE_ORIENTATION_2D


def set_default_image_interpolation(layer) -> None:
    for attribute in ("interpolation2d", "interpolation3d"):
        if hasattr(layer, attribute):
            setattr(layer, attribute, DEFAULT_INTERPOLATION)


def set_frame_axis_label(viewer) -> None:
    axis_labels = list(getattr(viewer.dims, "axis_labels", ()))
    if axis_labels:
        axis_labels[0] = "Frame"
        viewer.dims.axis_labels = tuple(axis_labels)


def update_frame_slider_labels(viewer, frame_index: int, frame_count: int) -> None:
    qt_viewer = getattr(getattr(viewer, "window", None), "_qt_viewer", None)
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
    total_label.setText(str(frame_count))
    width = current_label.fontMetrics().horizontalAdvance("8" * len(str(frame_count))) + 6
    current_label.setFixedWidth(width)
    total_label.setFixedWidth(width)


def hide_layer_controls(viewer) -> None:
    qt_viewer = getattr(getattr(viewer, "window", None), "_qt_viewer", None)
    controls_container = getattr(qt_viewer, "controls", None)
    controls_widgets = []
    controls = getattr(controls_container, "currentWidget", lambda: None)()
    if controls is not None:
        controls_widgets.append(controls)
    controls_widgets.extend(getattr(controls_container, "widgets", {}).values())
    for controls_widget in controls_widgets:
        _hide_widget_controls(controls_widget)


def _hide_widget_controls(controls) -> None:
    if controls is None:
        return
    transform_button = getattr(controls, "transform_button", None)
    if transform_button is not None:
        transform_button.setVisible(False)
    for control_name, widget_names in LAYER_CONTROLS_TO_HIDE.items():
        control = getattr(controls, control_name, None)
        if control is None:
            continue
        for widget_name in widget_names:
            widget = getattr(control, widget_name, None)
            if widget is not None:
                widget.setVisible(False)


def hide_viewer_mode_buttons(viewer) -> None:
    qt_viewer = getattr(getattr(viewer, "window", None), "_qt_viewer", None)
    viewer_buttons = getattr(qt_viewer, "viewerButtons", None)
    for button_name in ("ndisplayButton", "gridViewButton"):
        button = getattr(viewer_buttons, button_name, None)
        if button is not None:
            button.setVisible(False)

    qt_window = getattr(getattr(viewer, "window", None), "_qt_window", None)
    if qt_window is None:
        return
    for button in qt_window.findChildren(QAbstractButton):
        tooltip = button.toolTip()
        if "Toggle 2D/3D view" in tooltip or "Toggle grid mode" in tooltip:
            button.setVisible(False)
