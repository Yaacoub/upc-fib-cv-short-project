from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import os

import numpy as np

from .config import PipelineConfig


@dataclass
class Detection:
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    team_label: str | None = None
    dominant_color_rgb: tuple[int, int, int] | None = None
    dominant_colors_rgb: list[tuple[int, int, int]] | None = None
    track_id: int | None = None
    pitch_xy: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class YoloDetector:
    def __init__(self, config: PipelineConfig) -> None:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
        os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on local setup.
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.config = config
        self.model = YOLO(config.model)

    def detect(self, image_path: Path) -> list[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=1536,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        keep = {self.config.person_class, self.config.ball_class}
        # collect raw detections first
        raw = []
        for idx, (box, class_id, score) in enumerate(zip(xyxy, cls, conf)):
            if int(class_id) not in keep:
                continue
            x1, y1, x2, y2 = np.round(box).astype(int).tolist()
            raw.append((int(class_id), float(score), (int(x1), int(y1), int(x2), int(y2))))

        # image dims
        h_img = None
        w_img = None
        try:
            import cv2 as _cv

            img = _cv.imread(str(image_path))
            if img is not None:
                h_img, w_img = img.shape[:2]
        except Exception:
            pass

        # filter by confidence and size
        filtered: list[tuple[int, float, tuple[int, int, int, int]]] = []
        for class_id, score, (x1, y1, x2, y2) in raw:
            class_min_conf = (
                float(self.config.detection_filter.person_min_confidence)
                if class_id == self.config.person_class
                else float(self.config.detection_filter.ball_min_confidence)
            )
            if score < class_min_conf:
                continue
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            if w_img and h_img:
                area_ratio = (w * h) / float(w_img * h_img)
                min_area_ratio = (
                    float(self.config.detection_filter.min_box_area_ratio)
                    if class_id == self.config.person_class
                    else float(self.config.detection_filter.ball_min_box_area_ratio)
                )
                min_width = (
                    int(self.config.detection_filter.min_box_width)
                    if class_id == self.config.person_class
                    else int(self.config.detection_filter.ball_min_box_width)
                )
                min_height = (
                    int(self.config.detection_filter.min_box_height)
                    if class_id == self.config.person_class
                    else int(self.config.detection_filter.ball_min_box_height)
                )
                if area_ratio < min_area_ratio:
                    continue
                if area_ratio > float(self.config.detection_filter.max_box_area_ratio):
                    continue
            else:
                min_width = int(self.config.detection_filter.min_box_width)
                min_height = int(self.config.detection_filter.min_box_height)
            if w < min_width or h < min_height:
                continue
            filtered.append((class_id, score, (x1, y1, x2, y2)))

        # simple NMS
        if filtered:
            boxes_np = np.array([b for (_, _, b) in filtered], dtype=np.float32)
            scores = np.array([s for (_, s, _) in filtered], dtype=np.float32)
            x1 = boxes_np[:, 0]
            y1 = boxes_np[:, 1]
            x2 = boxes_np[:, 2]
            y2 = boxes_np[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]
            keep_indices: list[int] = []
            while order.size > 0:
                i = int(order[0])
                keep_indices.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w_int = np.maximum(0.0, xx2 - xx1 + 1)
                h_int = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w_int * h_int
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= float(self.config.iou))[0]
                order = order[inds + 1]

            for idx in keep_indices:
                class_id, score, (x1, y1, x2, y2) = filtered[int(idx)]
                detections.append(
                    Detection(
                        detection_id=len(detections),
                        class_id=int(class_id),
                        class_name=str(names.get(int(class_id), class_id)),
                        confidence=float(score),
                        bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )

        return detections


def person_detections(detections: Iterable[Detection], person_class: int) -> list[Detection]:
    return [d for d in detections if d.class_id == person_class]
