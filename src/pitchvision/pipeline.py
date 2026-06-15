from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .detection import YoloDetector
from .field_lines import detect_field_lines, detect_playing_field_mask
from .frames import list_images, FRAME_METADATA_FILENAME
from .projection import MiniMapConfig, estimate_pitch_homography, project_point
from .team_clustering import assign_team_clusters
from .tracking import SimpleTracker, bbox_bottom_center
from .visualization import (
    draw_checkpoint_overlay,
    draw_final_view,
    make_contact_sheet,
    write_video,
)


def _load_manual_labels(labels_path: Path | None) -> dict[tuple[str, int], str]:
    if labels_path is None or not labels_path.exists():
        return {}
    labels: dict[tuple[str, int], str] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[(row["image"], int(row["detection_id"]))] = row["label"]
    return labels


def _team_accuracy(
    records: list[dict],
    manual_labels: dict[tuple[str, int], str],
) -> dict:
    if not manual_labels:
        return {"available": False, "correct": 0, "total": 0, "accuracy": None}
    correct = 0
    total = 0
    for record in records:
        image_name = record["image"]
        for detection in record["detections"]:
            key = (image_name, detection["detection_id"])
            if key not in manual_labels:
                continue
            total += 1
            if detection.get("team_label") == manual_labels[key]:
                correct += 1
    return {
        "available": total > 0,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def write_manual_label_template(results_path: Path, output_path: Path) -> Path:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "detection_id", "suggested_label", "label", "bbox_xyxy"],
        )
        writer.writeheader()
        for record in results["records"]:
            for detection in record["detections"]:
                if detection["class_name"] != "person":
                    continue
                writer.writerow({
                    "image": record["image"],
                    "detection_id": detection["detection_id"],
                    "suggested_label": detection.get("team_label") or "",
                    "label": "",
                    "bbox_xyxy": detection["bbox_xyxy"],
                })
    return output_path


def _filter_detections_to_field(
    detections: list,
    field_rect: tuple[int, int, int, int] | None,
    field_mask,
) -> list:
    def inside_field(detection) -> bool:
        x1, y1, x2, y2 = detection.bbox_xyxy
        point = ((x1 + x2) // 2, y2)
        if field_mask is not None and field_mask.size:
            py, px = point[1], point[0]
            if py < 0 or px < 0 or py >= field_mask.shape[0] or px >= field_mask.shape[1]:
                return False
            return field_mask[py, px] > 0
        if field_rect is None:
            return True
        fx1, fy1, fx2, fy2 = field_rect
        return fx1 <= point[0] <= fx2 and fy1 <= point[1] <= fy2

    return [d for d in detections if d.class_name == "person" and inside_field(d)]


def run_checkpoint(
    image_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    manual_labels_path: Path | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    detection_label_dir = output_dir / "labels"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    detection_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    detector = YoloDetector(config)
    records: list[dict] = []
    overlay_paths: list[Path] = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        detections = detector.detect(image_path)
        field_mask_result = detect_playing_field_mask(image_bgr)
        field_rect = field_mask_result.rect_xyxy
        field_mask = field_mask_result.mask

        def _inside_field(detection) -> bool:
            x1, y1, x2, y2 = detection.bbox_xyxy
            point = ((x1 + x2) // 2, y2)
            if field_mask is not None and field_mask.size:
                py, px = point[1], point[0]
                if py < 0 or px < 0 or py >= field_mask.shape[0] or px >= field_mask.shape[1]:
                    return False
                return field_mask[py, px] > 0
            if field_rect is None:
                return True
            fx1, fy1, fx2, fy2 = field_rect
            return fx1 <= point[0] <= fx2 and fy1 <= point[1] <= fy2

        detections = [d for d in detections if d.class_name == "person" and _inside_field(d)]
        clusters = assign_team_clusters(
            image_bgr, detections, config.team_clustering, field_rect, field_mask
        )
        field_lines = detect_field_lines(image_bgr, config.field_lines, field_mask=field_mask)
        overlay = draw_checkpoint_overlay(image_bgr, detections, field_lines)
        overlay_path = overlay_dir / image_path.name
        cv2.imwrite(str(overlay_path), overlay)
        overlay_paths.append(overlay_path)

        record = {
            "image": image_path.name,
            "image_path": str(image_path),
            "overlay_path": str(overlay_path),
            "detections": [d.to_dict() for d in detections],
            "team_clusters": [c.__dict__ for c in clusters],
            "field_lines": field_lines.to_dict(),
        }
        records.append(record)
        label_path = detection_label_dir / f"{image_path.stem}.json"
        label_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    manual_labels = _load_manual_labels(manual_labels_path)
    person_counts = [
        sum(1 for d in record["detections"] if d["class_name"] == "person")
        for record in records
    ]
    ball_counts = [
        sum(1 for d in record["detections"] if d["class_name"] == "sports ball")
        for record in records
    ]
    summary = {
        "image_count": len(records),
        "total_person_detections": int(sum(person_counts)),
        "avg_person_detections": float(sum(person_counts) / len(person_counts)) if person_counts else 0.0,
        "total_ball_detections": int(sum(ball_counts)),
        "avg_field_lines": float(
            sum(record["field_lines"]["line_count"] for record in records) / len(records)
        ) if records else 0.0,
        "team_accuracy": _team_accuracy(records, manual_labels),
    }
    contact_sheet = make_contact_sheet(
        overlay_paths[:50], output_dir / "checkpoint_contact_sheet.jpg"
    )
    results = {
        "summary": summary,
        "records": records,
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results_path


def run_final_pipeline(
    image_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    manual_labels_path: Path | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    minimap_dir = output_dir / "minimap_frames"
    label_dir = output_dir / "labels"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    minimap_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    # Read frame rate from metadata written by extract-frames --all; fall back to 5 fps
    metadata_path = image_dir / FRAME_METADATA_FILENAME
    fps = 5.0
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fps = float(metadata.get("fps", 5.0))
        except (json.JSONDecodeError, ValueError):
            pass

    detector = YoloDetector(config)
    tracker = SimpleTracker()
    minimap_config = MiniMapConfig()
    records: list[dict] = []
    overlay_paths: list[Path] = []
    minimap_paths: list[Path] = []
    active_track_counts: list[int] = []

    camera_x_center = 52.5
    prev_track_positions: dict[int, float] = {}
    visible_width_m = config.projection.visible_pitch_width_m

    for frame_index, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        detections = detector.detect(image_path)
        field_mask_result = detect_playing_field_mask(image_bgr)
        field_rect = field_mask_result.rect_xyxy
        field_mask = field_mask_result.mask

        detections = _filter_detections_to_field(detections, field_rect, field_mask)
        clusters = assign_team_clusters(
            image_bgr,
            detections,
            config.team_clustering,
            field_rect,
            field_mask,
            camera_x_center=camera_x_center,
            visible_width_m=visible_width_m,
        )
        tracker.update(detections, image_bgr.shape)

        # Camera panning odometry: median player pixel displacement → meters
        matched_shifts: list[float] = []
        for detection in detections:
            if detection.class_name != "person" or detection.track_id is None:
                continue
            tid = detection.track_id
            curr_x = bbox_bottom_center(detection.bbox_xyxy)[0]
            if tid in prev_track_positions:
                matched_shifts.append(curr_x - prev_track_positions[tid])
        if matched_shifts:
            median_shift = float(np.median(matched_shifts))
            dx_meters = median_shift / image_bgr.shape[1] * visible_width_m
            camera_x_center = float(
                np.clip(
                    camera_x_center - dx_meters,
                    visible_width_m / 2.0,
                    105.0 - visible_width_m / 2.0,
                )
            )
        prev_track_positions = {
            d.track_id: bbox_bottom_center(d.bbox_xyxy)[0]
            for d in detections
            if d.class_name == "person" and d.track_id is not None
        }

        homography = estimate_pitch_homography(
            field_mask,
            image_bgr.shape,
            minimap_config,
            camera_x_center=camera_x_center,
            visible_width_m=visible_width_m,
        )
        for detection in detections:
            if detection.class_name != "person":
                continue
            detection.pitch_xy = project_point(
                bbox_bottom_center(detection.bbox_xyxy), homography
            )

        field_lines = detect_field_lines(image_bgr, config.field_lines, field_mask=field_mask)
        final_view = draw_final_view(image_bgr, detections, field_lines, minimap_config)

        overlay_path = overlay_dir / image_path.name
        minimap_path = minimap_dir / image_path.name
        cv2.imwrite(str(overlay_path), draw_checkpoint_overlay(image_bgr, detections, field_lines))
        cv2.imwrite(str(minimap_path), final_view)
        overlay_paths.append(overlay_path)
        minimap_paths.append(minimap_path)

        active_track_ids = sorted({
            d.track_id for d in detections
            if d.class_name == "person" and d.track_id is not None
        })
        active_track_counts.append(len(active_track_ids))

        record = {
            "frame_index": frame_index,
            "image": image_path.name,
            "image_path": str(image_path),
            "overlay_path": str(overlay_path),
            "minimap_path": str(minimap_path),
            "detections": [d.to_dict() for d in detections],
            "team_clusters": [c.__dict__ for c in clusters],
            "field_lines": field_lines.to_dict(),
            "homography": homography.to_dict(),
            "active_track_ids": active_track_ids,
        }
        records.append(record)
        (label_dir / f"{image_path.stem}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    manual_labels = _load_manual_labels(manual_labels_path)
    person_counts = [
        sum(1 for d in record["detections"] if d["class_name"] == "person")
        for record in records
    ]
    ball_counts = [
        sum(1 for d in record["detections"] if d["class_name"] == "sports ball")
        for record in records
    ]
    tracked_persons = [d for record in records for d in record["detections"]]
    projected_persons = [d for d in tracked_persons if d.get("pitch_xy") is not None]
    all_track_ids = {d["track_id"] for d in tracked_persons}

    summary = {
        "image_count": len(records),
        "total_person_detections": int(sum(person_counts)),
        "avg_person_detections": float(sum(person_counts) / len(person_counts)) if person_counts else 0.0,
        "total_ball_detections": int(sum(ball_counts)),
        "avg_field_lines": float(
            sum(record["field_lines"]["line_count"] for record in records) / len(records)
        ) if records else 0.0,
        "team_accuracy": _team_accuracy(records, manual_labels),
        "tracking": {
            "unique_track_ids": len(all_track_ids),
            "tracked_person_detections": len(tracked_persons),
            "projected_person_detections": len(projected_persons),
            "avg_active_tracks_per_frame": float(sum(active_track_counts) / len(active_track_counts)) if active_track_counts else 0.0,
            "created_tracks": tracker.created_tracks,
            "id_switches": None,
            "id_switch_note": "Ground-truth identities are unavailable for this subset, so true ID switches cannot be computed.",
        },
    }

    contact_sheet = make_contact_sheet(minimap_paths[:50], output_dir / "final_contact_sheet.jpg")
    video_path = write_video(minimap_paths, output_dir / "pitchvision_minimap.mp4", fps=fps)

    results = {
        "summary": summary,
        "records": records,
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
        "video": str(video_path) if video_path else None,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results_path