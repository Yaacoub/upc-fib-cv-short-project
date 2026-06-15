# PitchVision

[![Course](https://img.shields.io/badge/Course-Computer_Vision-blue.svg)](https://www.fib.upc.edu/en)
[![Institution](https://img.shields.io/badge/Institution-UPC_FIB-red.svg)](https://www.fib.upc.edu/en)

This repository contains the source code, report, and visual outputs for **PitchVision**, a football broadcast-to-mini-map projection pipeline.

The application detects players and the ball with YOLOv8, groups players by jersey color, extracts pitch markings, assigns simple frame-to-frame track IDs, estimates a pitch homography from the visible playing-field mask, and renders a side-by-side broadcast/minimap output. The final scope is intentionally KISS (Keep It Simple, Stupid) and documented in the [project report](doc/final_report/final_report.pdf).

## Key Features

### Zero-Shot Unsupervised Homography

No camera calibration parameters, SoccerNet homography files, or manual pitch annotations are used. The system estimates the visible pitch extent purely from the current frame's green-field mask and **camera panning odometry** (median player track displacement between frames). This makes PitchVision a Zero-Shot system that generalizes to unseen broadcasts without per-stadium metadata except for team colors.

### CLAHE and Saturation-Boosted Jersey Clustering

Each upper-body crop is enhanced in two stages before color clustering:
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) in LAB color space with a clip limit of 3.0
2. **Saturation/Value boosting** in HSV space (×1.8 / ×1.1)

This improves separation of similar jersey colors (e.g., blue vs. claret/purple) and differentiates the desaturated referee kit from the team kits.

### Goalkeeper and Referee Heuristics

After K-Means (K=2) clustering for the two main teams, **color outliers** are classified by their projected pitch position:
- **Goalkeepers** (near the goal lines, `x < 22m` or `x > 83m`) are associated with the defending team
- **Referees** (in the midfield region) are sorted by proximity to the center circle
- Population caps enforce domain rules: max 1 GK per side, max 2 referees, max 11 players per team

### Full-Match Output Video

`extract-frames --all` extracts every frame from the source broadcast video(s), in order, and records the source frame rate in `frame_metadata.json`. When `run` is pointed at that directory, `pitchvision_minimap.mp4` is encoded at the original broadcast frame rate, so the rendered output is a continuous full-match recording rather than a sparse 50-frame summary.

## System Requirements

> [!IMPORTANT]
> The commands below assume you are running them from the repository root.

The project targets a standard Python computer-vision environment on macOS.

*   **Python:** Python 3.9 or later.
*   **Model:** `yolov8n.pt` in the repository root, or any Ultralytics YOLO model path configured in `config/pipeline.yaml`.
*   **Libraries:** OpenCV, NumPy, scikit-learn, Pillow, PyYAML, Ultralytics, SoccerNet downloader dependencies.
*   **Optional:** `tectonic` to rebuild the LaTeX reports.

## Build and Run

### 1. Create the Environment

```bash
python3 -m venv .venv-pitchvision
. .venv-pitchvision/bin/activate
pip install -r requirements.txt
```

### 2. Download SoccerNet Videos

Before extracting frames, you can download the source broadcast videos. The downloader requires your SoccerNet NDA approval password. Set it in your environment or create a .env file in the repository root:

```bash
export SOCCERNET_PASSWORD="your_soccernet_password"
```

Run the download script to fetch the required game data (defaults to 720p resolution):
```bash
python scripts/download_data.py download-videos --resolution 720p
```

### 3. Extract Frames

The repository includes the 50-frame subset used for development and checkpointing in `data/subset/images/`. To regenerate it from local videos:

```bash
python scripts/pipeline.py extract-frames --videos data/soccernet --output data/subset/images --count 50
```

For the final pipeline run, extract every frame from the source video(s) instead, so the output video covers the first 30 seconds of the match:

```bash
python scripts/pipeline.py extract-frames --videos data/soccernet --output data/full_frames --all
```

This also writes `data/full_frames/frame_metadata.json`, which records the source video's frame rate so `run` can render the final video at the correct playback speed.

### 4. Run the 50% Checkpoint Pipeline

```bash
python scripts/pipeline.py run-checkpoint --images data/subset/images --output outputs/results
```

The checkpoint is intended for the 50-frame subset and produces only the contact sheet, overlays, and labels (no video).

### 5. Run the Complete Pipeline

```bash
python scripts/pipeline.py run --images data/full_frames --output outputs/final
```

This processes every extracted frame and writes:

*   `outputs/final/results.json`
*   `outputs/final/labels/*.json`
*   `outputs/final/overlays/*.jpg`
*   `outputs/final/minimap_frames/*.jpg`
*   `outputs/final/final_contact_sheet.jpg`
*   `outputs/final/pitchvision_minimap.mp4`

`pitchvision_minimap.mp4` is encoded using the frame rate recorded in `data/full_frames/frame_metadata.json`, so it plays back as a full-length match recording. If `--images` points at a directory with no `frame_metadata.json` (e.g. the 50-frame subset), the video falls back to 5 fps.

Running on every frame of a match processes far more images than the 50-frame subset, so expect this step to take significantly longer and use more disk space.

### 6. Run Tests

```bash
python -m unittest discover -s tests
```

## Configuration

All parameters can be tuned in `config/pipeline.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `team_clustering` | `contrast_clip_limit` | 3.0 | CLAHE clip limit for crop enhancement |
| `team_clustering` | `saturation_boost` | 1.8 | HSV saturation multiplier |
| `team_clustering` | `value_boost` | 1.1 | HSV value multiplier |
| `team_clustering` | `outlier_threshold` | 50.0 | Color distance threshold for outlier detection |
| `team_clustering` | `goalkeeper_distance_factor` | 1.35 | Scaling factor for goalkeeper color outlier detection |
| `team_clustering` | `max_players_per_team` | 11 | Maximum players per team (including GK) |
| `projection` | `visible_pitch_width_m` | 48.0 | Estimated visible horizontal pitch extent in meters |

## Controls

| Command | Action |
|---|---|
| `extract-frames` | Extract an evenly spaced frame subset from one video or a video directory |
| `extract-frames --all` | Extract every frame from the source video(s), in order, for a full-match final run |
| `run` | Execute detection, clustering, line extraction, tracking, projection, and mini-map rendering |
| `run-checkpoint` | Execute only the static-image checkpoint stages |
| `--manual-labels labels.csv` | Optionally evaluate team labels against manual annotations |
| `config/pipeline.yaml` | Change model path, confidence thresholds, team clustering, and line detection parameters |