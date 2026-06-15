from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass(frozen=True)
class HomographyResult:
    matrix: list[list[float]]
    source_points: list[tuple[float, float]]
    target_points: list[tuple[float, float]]
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MiniMapConfig:
    width_px: int = 520
    height_px: int = 340
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    margin_px: int = 28


def estimate_pitch_homography(
    field_mask: np.ndarray | None,
    image_shape: tuple[int, int, int],
    minimap: MiniMapConfig = MiniMapConfig(),
    camera_x_center: float = 52.5,
    visible_width_m: float = 48.0,
) -> HomographyResult:
    height, width = image_shape[:2]
    source = _source_quad_from_mask(field_mask, width, height)
    
    top_left, y_top = source[0]
    top_right, _ = source[1]
    bottom_right, y_bottom = source[2]
    bottom_left, _ = source[3]
    
    # Estimate the physical y-coordinate corresponding to the bottom of the image.
    # If the bottom edge of the green mask reaches the bottom of the image, the near touchline is off-screen.
    y_pitch_bottom = 68.0
    if field_mask is not None and field_mask.size and y_bottom >= height - 10:
        dy = y_bottom - y_top
        m_l = (bottom_left - top_left) / dy if dy > 0 else 0.0
        m_r = (bottom_right - top_right) / dy if dy > 0 else 0.0
        if m_r - m_l > 0.05:
            y_vanish = y_top - (top_right - top_left) / (m_r - m_l)
            R = 2.5
            y_pitch_bottom = 68.0 * (R / (R - 1.0)) * (y_bottom - y_top) / (y_bottom - y_vanish)
            y_pitch_bottom = min(68.0, max(20.0, y_pitch_bottom))

    x_start, x_end = _estimate_visible_pitch_bounds(field_mask, width, height, camera_x_center, visible_width_m)
    
    left_px = minimap.margin_px
    right_px = minimap.width_px - minimap.margin_px
    top_px = minimap.margin_px
    bottom_px = minimap.height_px - minimap.margin_px
    
    px_start = left_px + (x_start / 105.0) * (right_px - left_px)
    px_end = left_px + (x_end / 105.0) * (right_px - left_px)
    
    bottom_target_y = top_px + (y_pitch_bottom / 68.0) * (bottom_px - top_px)
    
    target = np.array(
        [
            [px_start, top_px],
            [px_end, top_px],
            [px_end, bottom_target_y],
            [px_start, bottom_target_y],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source.astype(np.float32), target)
    method = "field-mask row-span homography" if field_mask is not None and field_mask.size else "full-frame fallback homography"
    return HomographyResult(
        matrix=matrix.astype(float).tolist(),
        source_points=[tuple(map(float, row)) for row in source.tolist()],
        target_points=[tuple(map(float, row)) for row in target.tolist()],
        method=method,
    )


def _estimate_visible_pitch_bounds(
    field_mask: np.ndarray | None,
    width: int,
    height: int,
    camera_x_center: float,
    visible_width_m: float,
) -> tuple[float, float]:
    x_start = camera_x_center - visible_width_m / 2
    x_end = camera_x_center + visible_width_m / 2
    if x_start < 0.0:
        x_start = 0.0
        x_end = visible_width_m
    elif x_end > 105.0:
        x_end = 105.0
        x_start = 105.0 - visible_width_m
    return x_start, x_end


def project_point(point_xy: tuple[float, float], homography: HomographyResult) -> tuple[float, float]:
    matrix = np.array(homography.matrix, dtype=np.float32)
    point = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, matrix)[0, 0]
    return float(projected[0]), float(projected[1])


def _source_quad_from_mask(field_mask: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if field_mask is None or field_mask.size == 0 or int(field_mask.max()) == 0:
        return np.array(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
            ],
            dtype=np.float32,
        )

    ys, xs = np.where(field_mask > 0)
    if len(xs) < 10:
        return _source_quad_from_mask(None, width, height)

    y_top = int(np.percentile(ys, 5))
    y_bottom = int(np.percentile(ys, 95))

    def span_at(row_y: int) -> tuple[float, float]:
        band = max(4, height // 80)
        band_mask = (ys >= row_y - band) & (ys <= row_y + band)
        row_xs = xs[band_mask]
        if len(row_xs) < 2:
            return float(xs.min()), float(xs.max())
        return float(np.percentile(row_xs, 3)), float(np.percentile(row_xs, 97))

    top_left, top_right = span_at(y_top)
    bottom_left, bottom_right = span_at(y_bottom)

    # Broadcast views often show a trapezoid. Keep it convex and inside image.
    source = np.array(
        [
            [top_left, float(y_top)],
            [top_right, float(y_top)],
            [bottom_right, float(y_bottom)],
            [bottom_left, float(y_bottom)],
        ],
        dtype=np.float32,
    )
    source[:, 0] = np.clip(source[:, 0], 0, width - 1)
    source[:, 1] = np.clip(source[:, 1], 0, height - 1)
    return source
