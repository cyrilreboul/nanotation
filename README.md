# Nanotation

Nanotation is a standalone napari desktop application for annotating successive MRC image slices as one volume.

Package: `nanotation`

Launch command:

```bash
nanotation /path/to/mrc/slices
```

This app indexes individual MRC files from the selected folder as successive slices of one volume. It sorts numbered filenames naturally, opens at slice position zero, shows the folder as one scrollable napari volume, and exports per-slice point annotations to CSV. Slices are read only when viewed and recently viewed slices are cached; folder indexing also runs in the background so large datasets do not freeze the interface or fill memory at startup.

The viewer starts at `1.0x` zoom and includes `Zoom Out`, `Zoom 1:1`, and `Zoom In` controls.

Path checkpoints start with size `32`, border color `#0055ffff`, and opacity `0.6`. The right-side dock includes a read-only 3D scatter plot using the checkpoints' actual `(x, y, z)` coordinates. The VisPy canvas starts at `400x400`, shows a white volume bounding box and the current slice index, and highlights the current z-plane. Drag the plot to rotate it or scroll to zoom. A dashed path sorts checkpoints by slice index and connects only adjacent lower/higher-slice neighbors, so the first and last points have one connection and interior points have two.

Clicking a point in the 3D plot moves the main napari viewer to that point's slice. `Save Session…` writes a compact `.nanotation.json` file containing the source folder, annotations, coordinate scale, current slice, zoom, and 3D camera state. `Load Session…` reopens the source images lazily and restores the saved annotations and views.

Between the first and last annotated slices, the main image viewer displays a locked white cross at 40% opacity at the exact linear intersection of the current slice with the dashed 3D path. The image layer uses linear interpolation by default, and the napari interpolation selector is hidden when possible. The intersection is calculated on demand while scrolling, avoiding a large generated point layer for very large volumes. The right dock starts at a compact width and remains manually resizable.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Without an Initial Folder

```bash
nanotation
```

The app includes a folder picker when launched without a path.

## Volume Annotation CSV

`nanotation` exports these columns:

- `point_id`: sequential point number in the export
- `slice_index`: zero-based index of the nearest volume slice
- `filename`: MRC file corresponding to that slice
- `x`, `y`: napari data coordinates in pixels
- `xsc`, `ysc`: `x` and `y` multiplied by the `XY-Coordinate scale ouput` value

The CSV can also be exported when there are no points; it will contain only the header.
