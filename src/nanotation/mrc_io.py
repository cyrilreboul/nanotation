from __future__ import annotations

import os
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable

import mrcfile
import numpy as np

MRC_SUFFIXES = (".mrc", ".map", ".mrcs", ".mrc.gz", ".map.gz", ".mrcs.gz")
NATURAL_SORT_PATTERN = re.compile(r"(\d+)")


@dataclass(frozen=True, slots=True)
class MrcFrameRecord:
    """Metadata for one MRC file that contributes a frame to a time-series."""

    path: Path
    shape: tuple[int, int]
    dtype: str
    voxel_size: tuple[float, float, float] | None
    data_offset: int = 1024

    @property
    def name(self) -> str:
        return self.path.name


class MrcFrameSequence(Sequence[MrcFrameRecord]):
    """Compact sequence of records that shares time-series-wide metadata."""

    def __init__(
        self,
        folder: Path,
        relative_paths: Iterable[str],
        template: MrcFrameRecord,
    ) -> None:
        self._folder = folder
        self._relative_paths = tuple(relative_paths)
        self._shape = template.shape
        self._dtype = template.dtype
        self._voxel_size = template.voxel_size
        self._data_offset = template.data_offset

    def __len__(self) -> int:
        return len(self._relative_paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        return MrcFrameRecord(
            self._folder / self._relative_paths[index],
            self._shape,
            self._dtype,
            self._voxel_size,
            self._data_offset,
        )


def natural_sort_key(path: Path | str) -> tuple[object, ...]:
    """Sort numbered filenames in human order (frame2 before frame10)."""

    name = path.name if isinstance(path, Path) else path
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in NATURAL_SORT_PATTERN.split(name)
    )


def _mrc_filenames(folder: Path) -> list[str]:
    with os.scandir(folder) as entries:
        names = [
            entry.name
            for entry in entries
            if entry.is_file() and entry.name.lower().endswith(MRC_SUFFIXES)
        ]
    names.sort(key=natural_sort_key)
    return names


def scan_mrc_folder(
    folder: Path,
) -> Sequence[MrcFrameRecord]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a folder: {folder}")

    folder = folder.resolve()
    relative_paths = tuple(_mrc_filenames(folder))
    if not relative_paths:
        return []

    first = read_mrc_record(folder / relative_paths[0])
    return MrcFrameSequence(folder, relative_paths, first)


def read_mrc_record(path: Path) -> MrcFrameRecord:
    with mrcfile.open(path, mode="r", permissive=True, header_only=True) as mrc:
        raw_shape = tuple(int(value) for value in (mrc.header.nz, mrc.header.ny, mrc.header.nx))
        shape = _frame_shape(raw_shape, path)
        return MrcFrameRecord(
            path=path,
            shape=shape,
            dtype=str(np.dtype(mrcfile.utils.data_dtype_from_header(mrc.header))),
            voxel_size=_voxel_size_tuple(mrc.voxel_size),
            data_offset=1024 + int(mrc.header.nsymbt),
        )


def read_mrc_frame(
    path: Path,
    *,
    shape: tuple[int, int] | None = None,
    dtype: np.dtype | str | None = None,
    data_offset: int | None = None,
) -> np.ndarray:
    """Read one MRC file and require it to represent exactly one 2D frame."""

    if (
        shape is not None
        and dtype is not None
        and data_offset is not None
        and not path.name.lower().endswith(".gz")
    ):
        mapped = _memory_map_frame(path, shape, np.dtype(dtype), data_offset)
        if mapped is not None:
            return mapped

    opener = mrcfile.open if path.name.lower().endswith(".gz") else mrcfile.mmap
    with opener(path, mode="r", permissive=True) as mrc:
        data = np.asarray(mrc.data)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"Expected one 2D frame in {path.name}, found shape {data.shape}.")
        return data.copy()


class LazyMrcTimeSeries:
    """Array-like frame-y-x time-series that reads only requested MRC frames."""

    def __init__(self, records: Sequence[MrcFrameRecord], *, cache_size: int = 8) -> None:
        if not records:
            raise ValueError("No MRC files were found.")
        if cache_size < 1:
            raise ValueError("Cache size must be at least 1.")

        self._records = records
        self._shape = (len(records), *records[0].shape)
        self._dtype = np.dtype(records[0].dtype)
        self._cache_size = cache_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return 3

    @property
    def size(self) -> int:
        return int(np.prod(self._shape))

    def __getitem__(self, key):
        frame_key, y_key, x_key = _normalized_time_series_key(key)
        if isinstance(frame_key, Integral):
            image = self._read_frame(int(frame_key))
            return image[y_key, x_key]

        indices = range(*frame_key.indices(self.shape[0]))
        images = [self._read_frame(index)[y_key, x_key] for index in indices]
        if images:
            return np.stack(images, axis=0)

        sample_shape = np.empty(self.shape[1:], dtype=self.dtype)[y_key, x_key].shape
        return np.empty((0, *sample_shape), dtype=self.dtype)

    def _read_frame(self, index: int) -> np.ndarray:
        if index < 0:
            index += self.shape[0]
        if index < 0 or index >= self.shape[0]:
            raise IndexError("MRC time-series frame index out of range")

        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached

        record = self._records[index]
        image = read_mrc_frame(
            record.path,
            shape=record.shape,
            dtype=record.dtype,
            data_offset=record.data_offset,
        )
        if image.shape != self.shape[1:]:
            raise ValueError(
                f"All frames must have shape {self.shape[1:]}; "
                f"{record.name} has shape {image.shape}."
            )
        if image.dtype != self.dtype:
            raise ValueError(
                f"All frames must have dtype {self.dtype}; "
                f"{record.name} has dtype {image.dtype}."
            )

        self._cache[index] = image
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image


def _memory_map_frame(
    path: Path,
    shape: tuple[int, int],
    dtype: np.dtype,
    data_offset: int,
) -> np.ndarray | None:
    if data_offset < 1024:
        raise ValueError(f"Invalid MRC data offset in {path.name}: {data_offset}.")
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    try:
        raw = np.memmap(path, dtype=np.uint8, mode="r")
    except (OSError, ValueError):
        return None
    if raw.size < 1024 or bytes(raw[208:211]) != b"MAP":
        return None
    try:
        byte_order = mrcfile.utils.byte_order_from_machine_stamp(raw[212:216])
        header_values = np.frombuffer(raw, dtype=f"{byte_order}i4", count=24)
        actual_shape = tuple(int(value) for value in header_values[2::-1])
        actual_dtype = np.dtype(
            mrcfile.utils.dtype_from_mode(int(header_values[3]))
        ).newbyteorder(byte_order)
        actual_data_offset = 1024 + int(header_values[23])
    except (TypeError, ValueError):
        return None
    if (
        actual_shape != (1, *shape)
        or actual_dtype != dtype
        or actual_data_offset != data_offset
        or raw.size - data_offset != expected_bytes
    ):
        return None
    return raw[data_offset:].view(dtype).reshape(shape)


def _normalized_time_series_key(key) -> tuple[int | slice, int | slice, int | slice]:
    keys = key if isinstance(key, tuple) else (key,)
    if keys.count(Ellipsis) > 1:
        raise IndexError("Only one ellipsis is allowed in an index")
    if Ellipsis in keys:
        ellipsis_index = keys.index(Ellipsis)
        missing = 3 - (len(keys) - 1)
        keys = (
            *keys[:ellipsis_index],
            *(slice(None),) * missing,
            *keys[ellipsis_index + 1 :],
        )
    keys = (*keys, *(slice(None),) * (3 - len(keys)))
    if len(keys) != 3 or any(not isinstance(item, (Integral, slice)) for item in keys):
        raise IndexError(
            "MRC time-series data supports integer and range indexing in frame-y-x order"
        )
    return keys


def time_series_scale(records: Sequence[MrcFrameRecord]) -> tuple[float, float, float]:
    """Return napari z-y-x scale, using pixel units when metadata is absent."""

    if not records:
        return (1.0, 1.0, 1.0)
    voxel_size = records[0].voxel_size
    if voxel_size is None or any(value <= 0 for value in voxel_size):
        return (1.0, 1.0, 1.0)
    x_size, y_size, z_size = voxel_size
    return (z_size, y_size, x_size)


def _frame_shape(raw_shape: tuple[int, int, int], path: Path) -> tuple[int, int]:
    z_size, y_size, x_size = raw_shape
    if z_size != 1:
        raise ValueError(
            f"Expected each file to contain one 2D frame; {path.name} has shape {raw_shape}."
        )
    return y_size, x_size


def _voxel_size_tuple(voxel_size) -> tuple[float, float, float] | None:
    if voxel_size is None:
        return None
    values = tuple(float(value) for value in (voxel_size.x, voxel_size.y, voxel_size.z))
    if any(not np.isfinite(value) or value <= 0 for value in values):
        return None
    return values
