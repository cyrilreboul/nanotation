from pathlib import Path

import mrcfile
import numpy as np
import pytest

from nanotation.annotations import (
    DEFAULT_PATH_SMOOTHNESS,
    SmoothedPath,
    annotation_xyz_coordinates,
    dashed_neighbor_segments,
    frame_neighbor_edges,
    iter_annotation_entries,
    write_annotation_entries,
)
from nanotation.histogram import (
    finite_intensity_standard_deviation,
    standard_deviation_limits,
)
from nanotation.mrc_io import (
    LazyMrcTimeSeries,
    scan_mrc_folder,
    time_series_scale,
)
from nanotation.napari_ui import (
    DEFAULT_INTERPOLATION,
    EMAN2_IMAGE_ORIENTATION_2D,
    apply_eman2_image_orientation,
    set_default_image_interpolation,
)
from nanotation.plot3d import (
    CHECKPOINT_MARKER_SIZE,
    bounding_box_segments,
    homogeneous_canvas_positions,
    nearest_projected_point,
)
from nanotation.sessions import read_session_file, write_session_file
from nanotation.widgets import (
    IMAGE_CHECKPOINT_SIZE,
    _capture_refresh_annotations,
    _remap_refresh_annotations,
)


def _write_frame(path: Path, data: np.ndarray, voxel_size: float | None = None) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size


def test_scan_naturally_sorts_time_series(tmp_path: Path) -> None:
    _write_frame(tmp_path / "frame10.mrc", np.full((3, 4), 10, dtype=np.float32), 1.5)
    _write_frame(tmp_path / "frame2.mrc", np.full((3, 4), 2, dtype=np.float32), 1.5)
    _write_frame(tmp_path / "frame1.mrc", np.full((3, 4), 1, dtype=np.float32), 1.5)

    records = scan_mrc_folder(tmp_path)
    time_series = LazyMrcTimeSeries(records)

    assert [record.name for record in records] == ["frame1.mrc", "frame2.mrc", "frame10.mrc"]
    assert time_series.shape == (3, 3, 4)
    np.testing.assert_array_equal(time_series[:, 0, 0], [1, 2, 10])
    assert time_series_scale(records) == (1.5, 1.5, 1.5)


def test_time_series_rejects_mismatched_frame_shapes(tmp_path: Path) -> None:
    _write_frame(tmp_path / "frame1.mrc", np.zeros((3, 4), dtype=np.float32))
    _write_frame(tmp_path / "frame2.mrc", np.zeros((4, 4), dtype=np.float32))

    records = scan_mrc_folder(tmp_path)

    with pytest.raises(ValueError, match="All frames must have shape"):
        LazyMrcTimeSeries(records)[1]


def test_lazy_time_series_reads_only_requested_frames(tmp_path: Path, monkeypatch) -> None:
    for index in range(3):
        _write_frame(tmp_path / f"frame{index}.mrc", np.full((3, 4), index, dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    from nanotation import mrc_io

    original_read = mrc_io.read_mrc_frame
    reads = []

    def tracked_read(path: Path, **kwargs) -> np.ndarray:
        reads.append(path.name)
        return original_read(path, **kwargs)

    monkeypatch.setattr(mrc_io, "read_mrc_frame", tracked_read)
    time_series = LazyMrcTimeSeries(records, cache_size=2)

    np.testing.assert_array_equal(time_series[1], np.full((3, 4), 1, dtype=np.float32))
    np.testing.assert_array_equal(time_series[1, 1:, :2], np.full((2, 2), 1, dtype=np.float32))
    np.testing.assert_array_equal(time_series[1, ...], np.full((3, 4), 1, dtype=np.float32))
    np.testing.assert_array_equal(time_series[-1], np.full((3, 4), 2, dtype=np.float32))

    assert time_series.shape == (3, 3, 4)
    assert time_series.ndim == 3
    assert reads == ["frame1.mrc", "frame2.mrc"]


def test_lazy_time_series_uses_direct_memory_mapping(tmp_path: Path, monkeypatch) -> None:
    _write_frame(tmp_path / "frame0.mrc", np.arange(12, dtype=np.float32).reshape(3, 4))
    records = scan_mrc_folder(tmp_path)

    from nanotation import mrc_io

    def unexpected_mrc_open(*args, **kwargs):
        raise AssertionError("Uniform uncompressed frames should not be reparsed")

    monkeypatch.setattr(mrc_io.mrcfile, "mmap", unexpected_mrc_open)

    image = LazyMrcTimeSeries(records)[0]

    assert isinstance(image, np.memmap)
    np.testing.assert_array_equal(image, np.arange(12, dtype=np.float32).reshape(3, 4))


def test_lazy_time_series_falls_back_for_mismatched_layout(tmp_path: Path) -> None:
    _write_frame(tmp_path / "frame0.mrc", np.zeros((3, 4), dtype=np.float32))
    _write_frame(tmp_path / "frame1.mrc", np.zeros((4, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    with pytest.raises(ValueError, match="All frames must have shape"):
        LazyMrcTimeSeries(records)[1]


def test_lazy_time_series_rejects_same_size_mismatched_dtype(tmp_path: Path) -> None:
    _write_frame(tmp_path / "frame0.mrc", np.zeros((3, 4), dtype=np.float16))
    _write_frame(tmp_path / "frame1.mrc", np.zeros((3, 4), dtype=np.int16))
    records = scan_mrc_folder(tmp_path)

    with pytest.raises(ValueError, match="All frames must have dtype"):
        LazyMrcTimeSeries(records)[1]


def test_scan_reads_only_first_header(tmp_path: Path, monkeypatch) -> None:
    for index in range(3):
        _write_frame(tmp_path / f"frame{index}.mrc", np.zeros((3, 4), dtype=np.float32))

    from nanotation import mrc_io

    original_read = mrc_io.read_mrc_record
    reads = []

    def tracked_read(path: Path):
        reads.append(path.name)
        return original_read(path)

    monkeypatch.setattr(mrc_io, "read_mrc_record", tracked_read)

    records = scan_mrc_folder(tmp_path)

    assert len(records) == 3
    assert reads == ["frame0.mrc"]


def test_scan_does_not_create_index_files(tmp_path: Path) -> None:
    for index in range(3):
        _write_frame(tmp_path / f"frame{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    original_files = {path.name for path in tmp_path.iterdir()}

    scan_mrc_folder(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == original_files


def test_refresh_remaps_checkpoints_and_drops_deleted_files(tmp_path: Path) -> None:
    for frame_number in (1, 3, 5):
        _write_frame(
            tmp_path / f"frame{frame_number}.mrc",
            np.zeros((3, 4), dtype=np.float32),
        )
    old_records = scan_mrc_folder(tmp_path)
    refresh_annotations = _capture_refresh_annotations(
        np.array([[0, 1, 2], [1, 3, 4], [2, 5, 6]], dtype=float),
        old_records,
    )
    assert refresh_annotations is not None

    _write_frame(tmp_path / "frame0.mrc", np.zeros((3, 4), dtype=np.float32))
    (tmp_path / "frame3.mrc").unlink()
    new_records = scan_mrc_folder(tmp_path)

    remapped, removed_count = _remap_refresh_annotations(
        refresh_annotations,
        new_records,
    )

    np.testing.assert_array_equal(remapped, [[1, 1, 2], [2, 5, 6]])
    assert removed_count == 1


def test_refresh_does_not_remap_checkpoints_to_another_folder(tmp_path: Path) -> None:
    original_folder = tmp_path / "original"
    replacement_folder = tmp_path / "replacement"
    original_folder.mkdir()
    replacement_folder.mkdir()
    for folder in (original_folder, replacement_folder):
        _write_frame(folder / "frame1.mrc", np.zeros((3, 4), dtype=np.float32))

    refresh_annotations = _capture_refresh_annotations(
        np.array([[0, 1, 2]], dtype=float),
        scan_mrc_folder(original_folder),
    )
    assert refresh_annotations is not None

    remapped, removed_count = _remap_refresh_annotations(
        refresh_annotations,
        scan_mrc_folder(replacement_folder),
    )

    assert remapped.shape == (0, 3)
    assert removed_count == 1


def test_scan_rejects_files_containing_multiple_frames(tmp_path: Path) -> None:
    _write_frame(tmp_path / "stack.mrc", np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="one 2D frame"):
        scan_mrc_folder(tmp_path)


def test_eman2_image_orientation_sets_y_axis_up() -> None:
    class Camera:
        orientation2d = ("down", "right")

    class Viewer:
        camera = Camera()

    viewer = Viewer()
    apply_eman2_image_orientation(viewer)

    assert viewer.camera.orientation2d == EMAN2_IMAGE_ORIENTATION_2D


def test_default_image_interpolation_is_bicubic() -> None:
    class ImageLayer:
        interpolation2d = "linear"
        interpolation3d = "linear"

    layer = ImageLayer()

    set_default_image_interpolation(layer)

    assert DEFAULT_INTERPOLATION == "bicubic"
    assert layer.interpolation2d == "bicubic"
    assert layer.interpolation3d == "bicubic"


def test_default_checkpoint_sizes_are_distinct_for_image_and_3d_views() -> None:
    assert IMAGE_CHECKPOINT_SIZE == 32
    assert CHECKPOINT_MARKER_SIZE == 14


def test_finite_intensity_standard_deviation_ignores_nonfinite_values() -> None:
    values = np.array([np.nan, -1.0, 1.0, np.inf])

    assert finite_intensity_standard_deviation(values) == pytest.approx(1.0)
    assert finite_intensity_standard_deviation(np.array([np.nan, np.inf])) is None
    assert finite_intensity_standard_deviation(None) is None


def test_standard_deviation_limits() -> None:
    assert standard_deviation_limits(1.0, -3.0, 4.0) == (-3.0, 4.0)
    assert standard_deviation_limits(0.0, -3.0, 4.0) == (-1.0, 1.0)


def test_annotation_entries_include_intersection_and_checkpoint_frames(tmp_path: Path) -> None:
    for index in range(5):
        _write_frame(tmp_path / f"frame{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    entries = list(
        iter_annotation_entries(
            np.array([[0.0, 2.0, 4.0], [2.0, 6.0, 10.0]]),
            records,
        )
    )

    assert entries == [
        f"filename={tmp_path / 'frame0.mrc'} x=4.0 y=2.0 index=1 xdim=4 ydim=3",
        f"filename={tmp_path / 'frame1.mrc'} x=7.0 y=4.0 index=2 xdim=4 ydim=3",
        f"filename={tmp_path / 'frame2.mrc'} x=10.0 y=6.0 index=3 xdim=4 ydim=3",
    ]


def test_annotation_entries_export_single_checkpoint_frame(tmp_path: Path) -> None:
    for index in range(3):
        _write_frame(tmp_path / f"frame{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    entries = list(iter_annotation_entries(np.array([[2.0, 1.0, 2.0]]), records))

    assert entries == [
        f"filename={tmp_path / 'frame2.mrc'} x=2.0 y=1.0 index=3 xdim=4 ydim=3"
    ]


def test_annotation_entries_average_duplicate_checkpoint_frames(tmp_path: Path) -> None:
    for index in range(3):
        _write_frame(tmp_path / f"frame{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)

    entries = list(
        iter_annotation_entries(
            np.array([[1.0, 2.0, 4.0], [1.0, 6.0, 10.0]]),
            records,
        )
    )

    assert entries == [
        f"filename={tmp_path / 'frame1.mrc'} x=7.0 y=4.0 index=2 xdim=4 ydim=3"
    ]


def test_write_annotation_entries_supports_empty_and_populated_exports(tmp_path: Path) -> None:
    for index in range(3):
        _write_frame(tmp_path / f"frame{index}.mrc", np.zeros((3, 4), dtype=np.float32))
    records = scan_mrc_folder(tmp_path)
    output = tmp_path / "coordinates.txt"

    count = write_annotation_entries(output, np.empty((0, 3)), records)
    assert count == 0
    assert output.read_text(encoding="utf-8") == ""

    count = write_annotation_entries(
        output,
        np.array([[0, 1.0, 2.0], [2, 3.0, 6.0]]),
        records,
    )
    assert count == 3
    assert output.read_text(encoding="utf-8").splitlines() == [
        f"filename={tmp_path / 'frame0.mrc'} x=2.0 y=1.0 index=1 xdim=4 ydim=3",
        f"filename={tmp_path / 'frame1.mrc'} x=4.0 y=2.0 index=2 xdim=4 ydim=3",
        f"filename={tmp_path / 'frame2.mrc'} x=6.0 y=3.0 index=3 xdim=4 ydim=3",
    ]


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


def test_bounding_box_segments_use_full_xyz_extents() -> None:
    segments = bounding_box_segments(5, (3, 4))

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


def test_frame_neighbor_edges_connect_only_adjacent_z_points() -> None:
    xyz = np.array(
        [
            [10, 10, 62],
            [20, 20, 0],
            [30, 30, 70],
            [40, 40, 12],
        ],
        dtype=float,
    )

    edges = frame_neighbor_edges(xyz)

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


def test_path_intersection_interpolates_between_frame_neighbors() -> None:
    coordinates = np.array(
        [
            [10, 20, 40],
            [0, 0, 0],
            [20, 40, 80],
        ],
        dtype=float,
    )

    path = SmoothedPath(coordinates)

    np.testing.assert_allclose(path.at_frame(5), [5, 10, 20])
    np.testing.assert_allclose(path.at_frame(15), [15, 30, 60])
    np.testing.assert_allclose(path.at_frame(10), [10, 20, 40])
    assert path.at_frame(-1) is None
    assert path.at_frame(21) is None
    assert SmoothedPath(coordinates[:1]).at_frame(0) is None


def test_path_smoothness_reduces_measurement_spike() -> None:
    coordinates = np.array(
        [
            [0, 50, 50],
            [1, 50, 50],
            [2, 60, 50],
            [3, 50, 50],
            [4, 50, 50],
        ],
        dtype=float,
    )

    exact_path = SmoothedPath(coordinates, 0.0, (100, 100))
    mild_path = SmoothedPath(coordinates, DEFAULT_PATH_SMOOTHNESS, (100, 100))

    assert exact_path.at_frame(2)[1] == pytest.approx(60.0)
    assert 50.0 < mild_path.at_frame(2)[1] < 60.0
    assert mild_path.sample().shape == (65, 3)


def test_path_smoothness_rejects_values_outside_unit_range() -> None:
    coordinates = np.array([[0, 1, 1], [1, 2, 2]], dtype=float)

    with pytest.raises(ValueError, match="between 0 and 1"):
        SmoothedPath(coordinates, 1.01)


def test_session_file_round_trip(tmp_path: Path) -> None:
    for index in range(3):
        _write_frame(tmp_path / f"frame{index}.mrc", np.zeros((3, 4), dtype=np.float32))
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
        records=records,
        frame_number=3,
        zoom=1.25,
        camera_3d=camera,
        path_smoothness=0.35,
    )
    session = read_session_file(session_path)

    assert session.source_folder == tmp_path.resolve()
    np.testing.assert_array_equal(session.annotations, annotations)
    assert session.frame_count == 3
    assert session.first_filename == "frame0.mrc"
    assert session.last_filename == "frame2.mrc"
    assert session.frame_number == 3
    assert session.zoom == 1.25
    assert session.camera_3d == camera
    assert session.path_smoothness == pytest.approx(0.35)


def test_session_file_rejects_wrong_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"format": "something-else", "version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported Nanotation session"):
        read_session_file(path)


def test_session_file_rejects_frame_outside_saved_frame_count(tmp_path: Path) -> None:
    path = tmp_path / "bad-frame.json"
    path.write_text(
        '{"format":"nanotation-session","version":2,'
        '"source_folder":".","annotations_zyx":[],"image_count":1,'
        '"first_filename":"first.mrc","last_filename":"first.mrc",'
        '"frame_number":2,"zoom":1.0,"camera_3d":{}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid numeric session value"):
        read_session_file(path)


def test_legacy_session_converts_zero_based_index_to_frame_number(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"format":"nanotation-session","version":1,'
        '"source_folder":".","annotations_zyx":[],"image_count":1,'
        '"first_filename":"first.mrc","last_filename":"first.mrc",'
        '"slice_index":0,"zoom":1.0,"camera_3d":{}}',
        encoding="utf-8",
    )

    session = read_session_file(path)

    assert session.frame_number == 1
    assert session.path_smoothness == pytest.approx(DEFAULT_PATH_SMOOTHNESS)
