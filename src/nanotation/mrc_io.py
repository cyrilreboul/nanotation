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
class MrcSliceRecord:
    """Metadata for one MRC file that contributes a slice to a volume."""

    path: Path
    shape: tuple[int, int]
    dtype: str
    voxel_size: tuple[float, float, float] | None

    @property
    def name(self) -> str:
        return self.path.name


class MrcSliceCatalog(Sequence[MrcSliceRecord]):
    """Compact sequence of records that shares volume-wide metadata."""

    def __init__(self, paths: Iterable[Path], template: MrcSliceRecord) -> None:
        self._paths = tuple(paths)
        self._shape = template.shape
        self._dtype = template.dtype
        self._voxel_size = template.voxel_size

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        return MrcSliceRecord(
            self._paths[index],
            self._shape,
            self._dtype,
            self._voxel_size,
        )


def natural_sort_key(path: Path) -> tuple[object, ...]:
    """Sort numbered filenames in human order (slice2 before slice10)."""

    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in NATURAL_SORT_PATTERN.split(path.name)
    )


def is_mrc_path(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(MRC_SUFFIXES)


def iter_mrc_paths(folder: Path, *, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        paths = (path for path in folder.glob("**/*") if is_mrc_path(path))
    else:
        with os.scandir(folder) as entries:
            paths = [
                Path(entry.path)
                for entry in entries
                if entry.is_file() and entry.name.lower().endswith(MRC_SUFFIXES)
            ]
    yield from sorted(paths, key=natural_sort_key)


def scan_mrc_folder(folder: Path, *, recursive: bool = False) -> Sequence[MrcSliceRecord]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a folder: {folder}")

    paths = list(iter_mrc_paths(folder, recursive=recursive))
    if not paths:
        return []

    first = read_mrc_record(paths[0])
    return MrcSliceCatalog(paths, first)


def read_mrc_record(path: Path) -> MrcSliceRecord:
    with mrcfile.open(path, mode="r", permissive=True, header_only=True) as mrc:
        raw_shape = tuple(int(value) for value in (mrc.header.nz, mrc.header.ny, mrc.header.nx))
        shape = _slice_shape(raw_shape, path)
        mode = int(mrc.header.mode)
        return MrcSliceRecord(
            path=path,
            shape=shape,
            dtype=str(np.dtype(mrcfile.utils.dtype_from_mode(mode))),
            voxel_size=_voxel_size_tuple(mrc.voxel_size),
        )


def read_mrc_slice(path: Path) -> np.ndarray:
    """Read one MRC file and require it to represent exactly one 2D slice."""

    opener = mrcfile.open if path.name.lower().endswith(".gz") else mrcfile.mmap
    with opener(path, mode="r", permissive=True) as mrc:
        data = np.asarray(mrc.data)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"Expected one 2D slice in {path.name}, found shape {data.shape}.")
        return data.copy()


def load_mrc_volume(records: Sequence[MrcSliceRecord]) -> np.ndarray:
    """Load naturally ordered MRC records into a z-y-x volume."""

    if not records:
        raise ValueError("No MRC files were found.")

    expected_shape = records[0].shape
    slices = []
    for record in records:
        image = read_mrc_slice(record.path)
        if image.shape != expected_shape:
            raise ValueError(
                f"All slices must have shape {expected_shape}; "
                f"{record.name} has shape {image.shape}."
            )
        slices.append(image)
    return np.stack(slices, axis=0)


class LazyMrcVolume:
    """Array-like z-y-x volume that reads only requested MRC slices."""

    def __init__(self, records: Sequence[MrcSliceRecord], *, cache_size: int = 8) -> None:
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
        z_key, y_key, x_key = _normalized_volume_key(key)
        if isinstance(z_key, Integral):
            image = self._read_slice(int(z_key))
            return image[y_key, x_key]

        indices = range(*z_key.indices(self.shape[0]))
        images = [self._read_slice(index)[y_key, x_key] for index in indices]
        if images:
            return np.stack(images, axis=0)

        sample_shape = np.empty(self.shape[1:], dtype=self.dtype)[y_key, x_key].shape
        return np.empty((0, *sample_shape), dtype=self.dtype)

    def _read_slice(self, index: int) -> np.ndarray:
        if index < 0:
            index += self.shape[0]
        if index < 0 or index >= self.shape[0]:
            raise IndexError("MRC volume slice index out of range")

        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached

        record = self._records[index]
        image = read_mrc_slice(record.path)
        if image.shape != self.shape[1:]:
            raise ValueError(
                f"All slices must have shape {self.shape[1:]}; "
                f"{record.name} has shape {image.shape}."
            )
        if image.dtype != self.dtype:
            raise ValueError(
                f"All slices must have dtype {self.dtype}; "
                f"{record.name} has dtype {image.dtype}."
            )

        self._cache[index] = image
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image


def _normalized_volume_key(key) -> tuple[int | slice, int | slice, int | slice]:
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
        raise IndexError("MRC volumes support integer and slice indexing in z-y-x order")
    return keys


def volume_scale(records: Sequence[MrcSliceRecord]) -> tuple[float, float, float]:
    """Return napari z-y-x scale, using pixel units when metadata is absent."""

    if not records:
        return (1.0, 1.0, 1.0)
    voxel_size = records[0].voxel_size
    if voxel_size is None or any(value <= 0 for value in voxel_size):
        return (1.0, 1.0, 1.0)
    x_size, y_size, z_size = voxel_size
    return (z_size, y_size, x_size)


def _slice_shape(raw_shape: tuple[int, int, int], path: Path) -> tuple[int, int]:
    z_size, y_size, x_size = raw_shape
    if z_size != 1:
        raise ValueError(
            f"Expected each file to contain one 2D slice; {path.name} has shape {raw_shape}."
        )
    return y_size, x_size


def _voxel_size_tuple(voxel_size) -> tuple[float, float, float] | None:
    if voxel_size is None:
        return None
    values = tuple(float(value) for value in (voxel_size.x, voxel_size.y, voxel_size.z))
    if any(not np.isfinite(value) or value <= 0 for value in values):
        return None
    return values
