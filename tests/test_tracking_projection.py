from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchvision.detection import Detection
from pitchvision.projection import MiniMapConfig, estimate_pitch_homography, project_point
from pitchvision.tracking import SimpleTracker


def person(detection_id: int, bbox: tuple[int, int, int, int], team: str = "Team A") -> Detection:
    return Detection(
        detection_id=detection_id,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox_xyxy=bbox,
        team_label=team,
        dominant_color_rgb=(220, 40, 40) if team == "Team A" else (40, 80, 220),
    )


class TrackingTests(unittest.TestCase):
    def test_keeps_track_id_for_small_motion(self) -> None:
        tracker = SimpleTracker(max_missed=2, max_cost=0.4)
        first = [person(0, (100, 100, 130, 190))]
        second = [person(0, (106, 102, 136, 192))]

        tracker.update(first, (720, 1280, 3))
        first_id = first[0].track_id
        tracker.update(second, (720, 1280, 3))

        self.assertIsNotNone(first_id)
        self.assertEqual(second[0].track_id, first_id)
        self.assertEqual(tracker.created_tracks, 1)
        self.assertEqual(tracker.matched_detections, 1)

    def test_creates_new_track_for_large_jump(self) -> None:
        tracker = SimpleTracker(max_missed=2, max_cost=0.05)
        first = [person(0, (100, 100, 130, 190))]
        second = [person(0, (900, 400, 930, 500))]

        tracker.update(first, (720, 1280, 3))
        tracker.update(second, (720, 1280, 3))

        self.assertNotEqual(second[0].track_id, first[0].track_id)
        self.assertEqual(tracker.created_tracks, 2)


class ProjectionTests(unittest.TestCase):
    def test_projects_field_center_near_minimap_center(self) -> None:
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:90, 30:180] = 255
        minimap = MiniMapConfig(width_px=520, height_px=340, margin_px=28)

        homography = estimate_pitch_homography(mask, (100, 200, 3), minimap)
        projected = project_point((105.0, 55.0), homography)

        self.assertGreater(projected[0], 200)
        self.assertLess(projected[0], 320)
        self.assertGreater(projected[1], 130)
        self.assertLess(projected[1], 220)
        self.assertEqual(len(homography.matrix), 3)


if __name__ == "__main__":
    unittest.main()
