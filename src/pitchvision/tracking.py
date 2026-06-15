from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .detection import Detection


def bbox_bottom_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, float(y2))


@dataclass
class TrackState:
    track_id: int
    center_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    team_label: str | None
    color_rgb: tuple[int, int, int] | None
    missed_frames: int = 0
    age: int = 1

    def predicted_center(self) -> tuple[float, float]:
        return (
            self.center_xy[0] + self.velocity_xy[0],
            self.center_xy[1] + self.velocity_xy[1],
        )


class SimpleTracker:
    """Small MOT tracker for continuous broadcast shots.

    The original proposal mentioned YOLO feature maps, a Kalman filter, and the
    Hungarian algorithm. For this KISS implementation, each track uses a
    constant-velocity prediction, color/team appearance, and greedy assignment.
    This is deterministic, dependency-light, and enough for stable qualitative
    mini-map videos on the provided short frame subset.
    """

    def __init__(self, max_missed: int = 6, max_cost: float = 0.38) -> None:
        self.max_missed = max_missed
        self.max_cost = max_cost
        self._tracks: dict[int, TrackState] = {}
        self._next_id = 1
        self.created_tracks = 0
        self.matched_detections = 0

    @property
    def tracks(self) -> list[TrackState]:
        return sorted(self._tracks.values(), key=lambda track: track.track_id)

    def update(self, detections: list[Detection], frame_shape: tuple[int, int, int]) -> None:
        height, width = frame_shape[:2]
        scale = float(math.hypot(width, height))
        players = [d for d in detections if d.class_name == "person" and d.team_label is not None]

        candidates: list[tuple[float, int, int]] = []
        track_ids = list(self._tracks)
        for track_id in track_ids:
            track = self._tracks[track_id]
            predicted = track.predicted_center()
            for det_index, detection in enumerate(players):
                cost = self._assignment_cost(track, predicted, detection, scale)
                if cost <= self.max_cost:
                    candidates.append((cost, track_id, det_index))

        candidates.sort(key=lambda item: item[0])
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()

        for _, track_id, det_index in candidates:
            if track_id in assigned_tracks or det_index in assigned_detections:
                continue
            detection = players[det_index]
            self._assign_detection(track_id, detection)
            detection.track_id = track_id
            assigned_tracks.add(track_id)
            assigned_detections.add(det_index)
            self.matched_detections += 1

        for track_id in track_ids:
            if track_id in assigned_tracks:
                continue
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed:
                del self._tracks[track_id]

        for det_index, detection in enumerate(players):
            if det_index in assigned_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            center = bbox_bottom_center(detection.bbox_xyxy)
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                center_xy=center,
                velocity_xy=(0.0, 0.0),
                team_label=detection.team_label,
                color_rgb=detection.dominant_color_rgb,
            )
            detection.track_id = track_id
            self.created_tracks += 1

    def _assign_detection(self, track_id: int, detection: Detection) -> None:
        track = self._tracks[track_id]
        new_center = bbox_bottom_center(detection.bbox_xyxy)
        vx = new_center[0] - track.center_xy[0]
        vy = new_center[1] - track.center_xy[1]
        track.center_xy = new_center
        track.velocity_xy = (0.6 * vx + 0.4 * track.velocity_xy[0], 0.6 * vy + 0.4 * track.velocity_xy[1])
        track.team_label = detection.team_label
        track.color_rgb = detection.dominant_color_rgb or track.color_rgb
        track.missed_frames = 0
        track.age += 1

    @staticmethod
    def _assignment_cost(
        track: TrackState,
        predicted: tuple[float, float],
        detection: Detection,
        frame_scale: float,
    ) -> float:
        center = bbox_bottom_center(detection.bbox_xyxy)
        spatial = math.hypot(center[0] - predicted[0], center[1] - predicted[1]) / max(frame_scale, 1.0)
        team_penalty = 0.08 if track.team_label != detection.team_label else 0.0

        color_penalty = 0.0
        if track.color_rgb is not None and detection.dominant_color_rgb is not None:
            color_a = np.array(track.color_rgb, dtype=np.float32)
            color_b = np.array(detection.dominant_color_rgb, dtype=np.float32)
            color_penalty = min(0.18, float(np.linalg.norm(color_a - color_b)) / 255.0 * 0.18)

        return spatial + team_penalty + color_penalty
