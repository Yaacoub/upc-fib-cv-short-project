from __future__ import annotations

import json
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FRAME_METADATA_FILENAME = "frame_metadata.json"
DEFAULT_FPS = 25.0


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_videos(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return [path]
    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS
    )


def extract_evenly_spaced_frames(video_paths: list[Path], output_dir: Path, count: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if count <= 0:
        raise ValueError("Frame count must be positive.")
    if not video_paths:
        raise ValueError("No source videos were found.")

    captures = []
    total_frames = 0
    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        captures.append((video_path, capture, frames))
        total_frames += max(0, frames)

    if total_frames == 0:
        raise RuntimeError("The source videos do not report any readable frames.")

    saved_paths: list[Path] = []
    targets = [round(i * (total_frames - 1) / max(1, count - 1)) for i in range(count)]
    cumulative_start = 0

    for target_index, absolute_frame in enumerate(targets):
        for video_path, capture, frame_count in captures:
            if absolute_frame < cumulative_start + frame_count:
                local_frame = absolute_frame - cumulative_start
                capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                ok, frame = capture.read()
                if not ok:
                    break
                output_path = output_dir / f"frame_{target_index + 1:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_paths.append(output_path)
                break
            cumulative_start += frame_count
        cumulative_start = 0

    for _, capture, _ in captures:
        capture.release()

    return saved_paths


def extract_all_frames(video_paths: list[Path], output_dir: Path) -> list[Path]:
    """Extract every frame from the source videos, in order.

    Unlike `extract_evenly_spaced_frames`, this keeps every frame so the
    final pipeline run can stitch together an output video that covers the
    whole match instead of a sparse sample.

    The average source frame rate is written to `frame_metadata.json`
    alongside the images so `run_final_pipeline` can build the output video
    at the correct playback speed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not video_paths:
        raise ValueError("No source videos were found.")

    saved_paths: list[Path] = []
    fps_values: list[float] = []
    frame_index = 0

    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            fps_values.append(float(fps))

        # Limit extraction to the first 30 seconds
        max_frames = None
        if fps and fps > 0:
            max_frames = int(fps * 60 * 0.5)

        extracted_frames = 0

        while True:
            if max_frames is not None and extracted_frames >= max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            extracted_frames += 1

            output_path = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_paths.append(output_path)

        capture.release()

    if not saved_paths:
        raise RuntimeError("The source videos do not contain any readable frames.")

    metadata = {
        "fps": (sum(fps_values) / len(fps_values)) if fps_values else DEFAULT_FPS,
        "frame_count": len(saved_paths),
        "source_videos": [str(path) for path in video_paths],
    }
    (output_dir / FRAME_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return saved_paths


def read_frame_rate(image_dir: Path, default: float = DEFAULT_FPS) -> float:
    """Read the source frame rate written by `extract_all_frames`, if available.

    Falls back to `default` (used for evenly-spaced sample frames, which
    have no single meaningful playback rate).
    """
    metadata_path = image_dir / FRAME_METADATA_FILENAME
    if not metadata_path.exists():
        return default

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fps = float(metadata.get("fps", default))
        return fps if fps > 0 else default
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return default