import csv
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from nanotation.mrc_io import (
    LazyMrcVolume,
    load_mrc_volume,
    scan_mrc_folder,
    volume_scale,
)
from nanotation.widgets import (
    annotation_rows,
    annotation_xyz_coordinates,
    dashed_neighbor_segments,
    DEFAULT_INTERPOLATION,
    DEFAULT_OUTPUT_IMAGE_SIZE,
    EMAN2_IMAGE_ORIENTATION_2D,
    finite_intensity_range,
    finite_intensity_standard_deviation,
    homogeneous_canvas_positions,
    nearest_projected_point,
    path_intersection_at_slice,
    point_slice_summary,
    POINT_SIZE,
    read_session_file,
    slice_neighbor_edges,
    volume_box_segments,
    VolumeAnnotationWidget,
    write_session_file,
    write_annotations_csv,
)


def _write_slice(path: Path, data: np.ndarray, voxel_size: float | None = None) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size


def test_scan_naturally_sorts_and_loads_volume(tmp_path: Path) -> None:
    _write_slice(tmp_path / "slice10.mrc", np.full((3, 4), 10, dtype=np.float32), 1.5)
    _write_slice(tmp_path / "slice2.mrc", np.full((3, 4), 2, dtype=np.float32), 1.5)
    _write_slice(tmp_path / "slice1.mrc", np.full((3, 4), 1, dtype=np.float32), 1.5)

    records = scan_mrc_folder(tmp_path)
    volume = load_mrc_volume(records)

    assert [record.name for record in records] == ["slice1.mrc", "slice2.mrc", "slice10.mrc"]
    assert volume.shape == (3, 3, 4)
    np.testing.assert_array_equal(volume[:, 0, 0], [1, 2, 10])
    assert volume_scale(records) == (1.5, 1.5, 1.5)


def test_volume_rejects_mismatched_slice_shapes(tmp_path: Path) -> None:
    _write_slice(tmp_path / "slice1.mrc", np.zeros((3, 4), dtype=np.float32))
    _write_slice(tmp_path / "slice2.mrc", np.zeros((4, 4), dtype=np.float32))

    records = scan_mrc_folder(tmp_path)

    with pytest.raises(ValueError, match="All slices must have shape"):
        load_mrc_volume(records)


def test_lazy_volume_reads_only_requested_slices(tmp_path: Path, monkeypatch) -> None:
    for index in range(3):
        _write_slice(tmp_path / f"slice{index}.mrc", np.full((3, 4), index, dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    from nanotation import mrc_io

    original_read = mrc_io.read_mrc_slice
    reads = []

    def tracked_read(path: Path) -> np.ndarray:
        reads.append(path.name)
        return original_read(path)

    monkeypatch.setattr(mrc_io, "read_mrc_slice", tracked_read)
    volume = LazyMrcVolume(records, cache_size=2)

    np.testing.assert_array_equal(volume[1], np.full((3, 4), 1, dtype=np.float32))
    np.testing.assert_array_equal(volume[1, 1:, :2], np.full((2, 2), 1, dtype=np.float32))
    np.testing.assert_array_equal(volume[1, ...], np.full((3, 4), 1, dtype=np.float32))
    np.testing.assert_array_equal(volume[-1], np.full((3, 4), 2, dtype=np.float32))

    assert volume.shape == (3, 3, 4)
    assert volume.ndim == 3
    assert reads == ["slice1.mrc", "slice2.mrc"]


def test_scan_reads_only_first_header(tmp_path: Path, monkeypatch) -> None:
    for index in range(3):
        _write_slice(tmp_path / f"slice{index}.mrc", np.zeros((3, 4), dtype=np.float32))

    from nanotation import mrc_io

    original_read = mrc_io.read_mrc_record
    reads = []

    def tracked_read(path: Path):
        reads.append(path.name)
        return original_read(path)

    monkeypatch.setattr(mrc_io, "read_mrc_record", tracked_read)

    records = scan_mrc_folder(tmp_path)

    assert len(records) == 3
    assert reads == ["slice0.mrc"]


def test_scan_rejects_files_containing_multiple_slices(tmp_path: Path) -> None:
    _write_slice(tmp_path / "stack.mrc", np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="one 2D slice"):
        scan_mrc_folder(tmp_path)


def test_default_output_image_size_tracks_point_size() -> None:
    assert DEFAULT_OUTPUT_IMAGE_SIZE == int(POINT_SIZE * 1.5)


def test_eman2_image_orientation_sets_y_axis_up() -> None:
    class Camera:
        orientation2d = ("down", "right")

    class Viewer:
        camera = Camera()

    class Widget:
        viewer = Viewer()

    VolumeAnnotationWidget._apply_eman2_image_orientation(Widget())

    assert Widget.viewer.camera.orientation2d == EMAN2_IMAGE_ORIENTATION_2D


def test_default_image_interpolation_is_bicubic() -> None:
    class ImageLayer:
        interpolation2d = "linear"
        interpolation3d = "linear"

    layer = ImageLayer()

    VolumeAnnotationWidget._set_default_interpolation(None, layer)

    assert DEFAULT_INTERPOLATION == "bicubic"
    assert layer.interpolation2d == "bicubic"
    assert layer.interpolation3d == "bicubic"


def test_finite_intensity_range_ignores_nonfinite_values() -> None:
    assert finite_intensity_range(np.array([np.nan, -2.0, 4.0, np.inf])) == (-2.0, 4.0)
    assert finite_intensity_range(np.array([np.nan, np.inf])) is None


def test_finite_intensity_standard_deviation_ignores_nonfinite_values() -> None:
    values = np.array([np.nan, -1.0, 1.0, np.inf])

    assert finite_intensity_standard_deviation(values) == pytest.approx(1.0)
    assert finite_intensity_standard_deviation(np.array([np.nan, np.inf])) is None


def test_annotation_rows_include_intersection_and_point_slices(tmp_path: Path) -> None:
    for index in range(5):
        _write_slice(tmp_path / f"slice{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    rows = annotation_rows(
        np.array([[0.0, 2.0, 4.0], [2.0, 6.0, 10.0]]),
        records,
        coordinate_scale=4,
    )

    assert [row["slice_index"] for row in rows] == [0, 1, 2]
    assert [row["filename"] for row in rows] == ["slice0.mrc", "slice1.mrc", "slice2.mrc"]
    assert rows[1] == {
        "filename": "slice1.mrc",
        "slice_index": 1,
        "x": 7.0,
        "y": 4.0,
        "xsc": 28.0,
        "ysc": 16.0,
    }


def test_annotation_rows_export_single_checkpoint_slice(tmp_path: Path) -> None:
    for index in range(3):
        _write_slice(tmp_path / f"slice{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    rows = annotation_rows(np.array([[2.0, 1.0, 2.0]]), records, coordinate_scale=4)

    assert rows == [
        {
            "filename": "slice2.mrc",
            "slice_index": 2,
            "x": 2.0,
            "y": 1.0,
            "xsc": 8.0,
            "ysc": 4.0,
        }
    ]


def test_annotation_rows_average_duplicate_point_slices(tmp_path: Path) -> None:
    for index in range(3):
        _write_slice(tmp_path / f"slice{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    rows = annotation_rows(
        np.array([[1.0, 2.0, 4.0], [1.0, 6.0, 10.0]]),
        records,
        coordinate_scale=2,
    )

    assert rows == [
        {
            "filename": "slice1.mrc",
            "slice_index": 1,
            "x": 7.0,
            "y": 4.0,
            "xsc": 14.0,
            "ysc": 8.0,
        }
    ]


def test_write_annotations_csv_supports_empty_and_populated_exports(tmp_path: Path) -> None:
    for index in range(3):
        _write_slice(tmp_path / f"slice{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)
    output = tmp_path / "coordinates.csv"

    count = write_annotations_csv(output, np.empty((0, 3)), records)
    assert count == 0
    assert output.read_text(encoding="utf-8").strip() == "filename,slice_index,x,y,xsc,ysc"

    count = write_annotations_csv(
        output,
        np.array([[0, 1.0, 2.0], [2, 3.0, 6.0]]),
        records,
        coordinate_scale=4,
    )
    with output.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    assert count == 3
    assert [row["filename"] for row in rows] == ["slice0.mrc", "slice1.mrc", "slice2.mrc"]
    assert rows[1]["slice_index"] == "1"
    assert rows[1]["y"] == "2.0"
    assert rows[1]["x"] == "4.0"
    assert rows[1]["ysc"] == "8.0"
    assert rows[1]["xsc"] == "16.0"


def test_annotation_export_rejects_invalid_coordinate_scale(tmp_path: Path) -> None:
    _write_slice(tmp_path / "slice0.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    with pytest.raises(ValueError, match="Coordinate scale"):
        annotation_rows(np.array([[0, 1.25, 2.5]]), records, coordinate_scale=0)


def test_point_slice_summary_groups_points_by_slice(tmp_path: Path) -> None:
    for index in range(3):
        _write_slice(tmp_path / f"slice{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    summary = point_slice_summary(np.array([[0, 1, 1], [2.1, 2, 2], [1.9, 3, 3]]), records)

    assert summary == "Annotated slices: 0 (slice0.mrc): 1 point; 2 (slice2.mrc): 2 points"


def test_annotation_xyz_coordinates_converts_zyx_to_xyz() -> None:
    coordinates = np.array(
        [
            [0.0, 10.0, 20.0],
            [2.0, 30.0, 40.0],
            [np.nan, 50.0, 60.0],
        ]
    )

    np.testing.assert_array_equal(
        annotation_xyz_coordinates(coordinates),
        [[20.0, 10.0, 0.0], [40.0, 30.0, 2.0]],
    )


def test_volume_box_segments_use_full_xyz_extents() -> None:
    segments = volume_box_segments(5, (3, 4))

    assert segments.shape == (24, 3)
    np.testing.assert_array_equal(segments.min(axis=0), [0, 0, 0])
    np.testing.assert_array_equal(segments.max(axis=0), [3, 2, 4])


def test_nearest_projected_point_uses_click_threshold() -> None:
    positions = np.array([[10, 20], [40, 50], [100, 120]], dtype=float)

    assert nearest_projected_point(positions, (43, 54)) == 1
    assert nearest_projected_point(positions, (70, 80)) is None
    assert nearest_projected_point(np.empty((0, 2)), (10, 20)) is None


def test_homogeneous_canvas_positions_divides_by_weight() -> None:
    projected = np.array(
        [
            [200, 400, 10, 2],
            [300, 150, 20, 3],
            [10, 20, 30, 0],
        ],
        dtype=float,
    )

    positions = homogeneous_canvas_positions(projected)

    np.testing.assert_array_equal(positions[:2], [[100, 200], [100, 50]])
    assert np.isinf(positions[2]).all()


def test_slice_neighbor_edges_connect_only_adjacent_z_points() -> None:
    xyz = np.array(
        [
            [10, 10, 62],
            [20, 20, 0],
            [30, 30, 70],
            [40, 40, 12],
        ],
        dtype=float,
    )

    edges = slice_neighbor_edges(xyz)

    np.testing.assert_array_equal(edges[:, :, 2], [[0, 12], [12, 62], [62, 70]])
    assert np.count_nonzero(edges[:, :, 2] == 0) == 1
    assert np.count_nonzero(edges[:, :, 2] == 70) == 1
    assert np.count_nonzero(edges[:, :, 2] == 12) == 2
    assert np.count_nonzero(edges[:, :, 2] == 62) == 2


def test_dashed_neighbor_segments_leave_regular_gaps() -> None:
    xyz = np.array([[0, 0, 0], [100, 0, 10]], dtype=float)

    segments = dashed_neighbor_segments(xyz, dash_count=4, dash_fraction=0.5)

    assert segments.shape == (8, 3)
    np.testing.assert_allclose(segments[:, 0], [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5])
    np.testing.assert_allclose(segments[:, 2], [0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75])


def test_path_intersection_interpolates_between_slice_neighbors() -> None:
    coordinates = np.array(
        [
            [10, 20, 40],
            [0, 0, 0],
            [20, 40, 80],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(path_intersection_at_slice(coordinates, 5), [5, 10, 20])
    np.testing.assert_allclose(path_intersection_at_slice(coordinates, 15), [15, 30, 60])
    np.testing.assert_allclose(path_intersection_at_slice(coordinates, 10), [10, 20, 40])
    assert path_intersection_at_slice(coordinates, -1) is None
    assert path_intersection_at_slice(coordinates, 21) is None
    assert path_intersection_at_slice(coordinates[:1], 0) is None


def test_session_file_round_trip(tmp_path: Path) -> None:
    for index in range(3):
        _write_slice(tmp_path / f"slice{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)
    session_path = tmp_path / "work.nanotation.json"
    annotations = np.array([[0, 1.5, 2.5], [2, 2.0, 3.0]], dtype=float)
    camera = {
        "azimuth": 35.0,
        "elevation": 25.0,
        "roll": 0.0,
        "scale_factor": 12.0,
        "center": [1.5, 1.0, 1.0],
    }

    write_session_file(
        session_path,
        source_folder=tmp_path,
        annotations=annotations,
        coordinate_scale=4,
        records=records,
        slice_index=2,
        zoom=1.25,
        camera_3d=camera,
    )
    session = read_session_file(session_path)

    assert session.source_folder == tmp_path.resolve()
    np.testing.assert_array_equal(session.annotations, annotations)
    assert session.coordinate_scale == 4
    assert session.image_count == 3
    assert session.first_filename == "slice0.mrc"
    assert session.last_filename == "slice2.mrc"
    assert session.slice_index == 2
    assert session.zoom == 1.25
    assert session.camera_3d == camera


def test_session_file_rejects_wrong_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"format": "something-else", "version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported Nanotation session"):
        read_session_file(path)
