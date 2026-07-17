# Nanotation

## Precision annotation for large MRC time-series

Nanotation is a focused desktop application for tracing structures and motion across successive electron-microscopy frames. It presents a folder of individual MRC images as one continuous time-series, allowing you to move naturally through the data and place precise checkpoints wherever they are needed.

As checkpoints are added, Nanotation builds a connected spatial path and displays it in an interactive 3D overview. The path is projected through the frames between checkpoints, making its progression easy to follow without requiring a point to be placed manually in every image.

Nanotation is designed to remain responsive with exceptionally large datasets. Images are opened only as they are viewed, so long time-series can be explored without loading the entire collection into memory.

## Highlights

- Seamless navigation through naturally ordered MRC frames
- Precise checkpoint placement with EMAN2-style image orientation
- Smooth path estimation between annotated frames
- Adjustable path smoothing for treating checkpoints as experimental measurements
- Interactive, rotatable 3D overview of the complete annotated path
- Direct navigation from a 3D checkpoint to its corresponding frame
- Adjustable contrast through a compact interactive histogram
- Session saving and restoration for long-running annotation work
- Clean CSV export for downstream analysis

## Typical Workflow

1. Select a folder containing the MRC frames of a time-series.
2. Move through the frames using the frame control.
3. Place checkpoints at locations of interest.
4. Adjust **Path Smoothness** between `0.00` and `1.00` if the measured checkpoints require more or less smoothing.
5. Review the resulting path in the 3D overview and navigate by clicking its checkpoints.
6. Save the session at any time or export the frame-by-frame coordinates.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Launch

Open Nanotation with a time-series folder:

```bash
nanotation /path/to/mrc/frames
```

Alternatively, launch it without a path and choose a folder from the application:

```bash
nanotation
```

## Sessions

Use **Save Session…** to preserve the source folder, checkpoints, path smoothness, current frame, zoom level, and 3D viewpoint. **Load Session…** restores the workspace so annotation can continue from the same point.

## Coordinate Export

Nanotation exports one CSV row for each frame containing either a checkpoint or a smoothed path position:

| Column | Description |
|---|---|
| `filename` | MRC file associated with the frame |
| `frame_number` | One-based frame number |
| `x`, `y` | Base-zero image coordinates, measured from the bottom-left origin |

The exported path is ready for analysis, visualization, or integration into a larger processing workflow.
