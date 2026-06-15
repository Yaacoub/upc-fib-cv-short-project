from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detection import Detection
from .field_lines import FieldLineResult
from .projection import MiniMapConfig


TEAM_COLORS_BGR = {
    "Team A": (255, 120, 35),        # Blue (BGR: Blue=255, Green=120, Red=35)
    "Team B": (45, 90, 255),         # Orange/Red (BGR: Blue=45, Green=90, Red=255)
    "Goalkeeper A": (50, 220, 50),   # Neon Green
    "Goalkeeper B": (220, 50, 220),  # Neon Pink/Purple
    "Referee": (0, 240, 240),        # Neon Yellow/Cyan
    None: (220, 220, 220),
}


def draw_checkpoint_overlay(
    image_bgr: np.ndarray,
    detections: list[Detection],
    field_lines: FieldLineResult,
) -> np.ndarray:
    output = image_bgr.copy()

    for x1, y1, x2, y2 in field_lines.lines_xyxy:
        cv2.line(output, (x1, y1), (x2, y2), (80, 255, 80), 2, cv2.LINE_AA)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = TEAM_COLORS_BGR.get(detection.team_label, (220, 220, 220))
        label = detection.team_label or detection.class_name
        if detection.class_name != "person":
            color = (0, 230, 255)
            label = detection.class_name
        track = f"T{detection.track_id} " if detection.track_id is not None else ""
        text = f"{track}{detection.detection_id}: {label} {detection.confidence:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, max(0, y1 - th - baseline - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            output,
            text,
            (x1 + 3, max(12, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_minimap(
    detections: list[Detection],
    config: MiniMapConfig = MiniMapConfig(),
) -> np.ndarray:
    canvas = np.full((config.height_px, config.width_px, 3), (42, 118, 70), dtype=np.uint8)
    line = (238, 238, 230)
    margin = config.margin_px
    left, top = margin, margin
    right, bottom = config.width_px - margin, config.height_px - margin
    mid_x = (left + right) // 2
    mid_y = (top + bottom) // 2

    cv2.rectangle(canvas, (left, top), (right, bottom), line, 2)
    cv2.line(canvas, (mid_x, top), (mid_x, bottom), line, 2)
    cv2.circle(canvas, (mid_x, mid_y), 36, line, 2, cv2.LINE_AA)
    cv2.circle(canvas, (mid_x, mid_y), 3, line, -1, cv2.LINE_AA)

    box_w = int((right - left) * 0.16)
    box_h = int((bottom - top) * 0.42)
    cv2.rectangle(canvas, (left, mid_y - box_h // 2), (left + box_w, mid_y + box_h // 2), line, 2)
    cv2.rectangle(canvas, (right - box_w, mid_y - box_h // 2), (right, mid_y + box_h // 2), line, 2)

    for detection in detections:
        if detection.class_name != "person" or detection.pitch_xy is None:
            continue
        x, y = detection.pitch_xy
        px = int(np.clip(round(x), left, right))
        py = int(np.clip(round(y), top, bottom))
        color = TEAM_COLORS_BGR.get(detection.team_label, (220, 220, 220))
        cv2.circle(canvas, (px, py), 8, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 8, (20, 25, 22), 1, cv2.LINE_AA)
        if detection.track_id is not None:
            label = str(detection.track_id)
            cv2.putText(
                canvas,
                label,
                (px + 10, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (250, 250, 245),
                1,
                cv2.LINE_AA,
            )

    return canvas


def draw_final_view(
    image_bgr: np.ndarray,
    detections: list[Detection],
    field_lines: FieldLineResult,
    config: MiniMapConfig = MiniMapConfig(),
) -> np.ndarray:
    overlay = draw_checkpoint_overlay(image_bgr, detections, field_lines)
    minimap = draw_minimap(detections, config)
    target_h = overlay.shape[0]
    scale = target_h / minimap.shape[0]
    minimap_resized = cv2.resize(minimap, (int(minimap.shape[1] * scale), target_h))
    return np.hstack([overlay, minimap_resized])


def write_video(frame_paths: list[Path], output_path: Path, fps: float = 5.0) -> Path | None:
    if not frame_paths:
        return None
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return None
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    return output_path if output_path.exists() else None


def make_contact_sheet(image_paths: list[Path], output_path: Path, thumb_width: int = 320) -> Path | None:
    if not image_paths:
        return None

    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = thumb_width / max(1, w)
        thumb = cv2.resize(image, (thumb_width, max(1, int(h * scale))))
        thumbs.append(thumb)

    if not thumbs:
        return None

    cols = min(5, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    thumb_height = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * thumb_height, cols * thumb_width, 3), 245, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        y = row * thumb_height
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return output_path
