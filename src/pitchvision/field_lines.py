from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .config import FieldLineConfig


@dataclass(frozen=True)
class FieldLineResult:
    line_count: int
    lines_xyxy: list[tuple[int, int, int, int]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FieldMaskResult:
    mask: np.ndarray
    rect_xyxy: tuple[int, int, int, int] | None


def detect_playing_field_mask(image_bgr: np.ndarray) -> FieldMaskResult:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # green field segmentation to reject stands and billboards
    green_mask = ((hue >= 35) & (hue <= 95) & (sat > 35) & (val > 35)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        empty = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        return FieldMaskResult(mask=empty, rect_xyxy=None)

    h_img, w_img = image_bgr.shape[:2]
    min_area = 0.05 * float(h_img * w_img)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        empty = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        return FieldMaskResult(mask=empty, rect_xyxy=None)

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    x, y, w, h = cv2.boundingRect(largest)
    rect = (int(x), int(y), int(x + w), int(y + h))
    return FieldMaskResult(mask=mask, rect_xyxy=rect)


def detect_field_lines(
    image_bgr: np.ndarray,
    config: FieldLineConfig,
    field_mask: np.ndarray | None = None,
) -> FieldLineResult:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    white_mask = (saturation < 60) & (value > 185)
    mask = (white_mask.astype(np.uint8) * 255)
    if field_mask is not None and field_mask.size:
        mask = cv2.bitwise_and(mask, field_mask)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    edges = cv2.Canny(mask, config.canny_low, config.canny_high)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.min_line_length,
        maxLineGap=config.max_line_gap,
    )

    lines: list[tuple[int, int, int, int]] = []
    if raw_lines is not None:
        for line in raw_lines[:, 0, :]:
            x1, y1, x2, y2 = (int(v) for v in line.tolist())
            if field_mask is not None and field_mask.size:
                samples = np.linspace(0.0, 1.0, 9)
                xs = np.round(x1 + (x2 - x1) * samples).astype(int)
                ys = np.round(y1 + (y2 - y1) * samples).astype(int)
                xs = np.clip(xs, 0, field_mask.shape[1] - 1)
                ys = np.clip(ys, 0, field_mask.shape[0] - 1)
                if float(field_mask[ys, xs].mean()) < 120:
                    continue
            lines.append((x1, y1, x2, y2))

    return FieldLineResult(line_count=len(lines), lines_xyxy=lines)
