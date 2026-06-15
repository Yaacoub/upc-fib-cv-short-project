from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchvision.config import PipelineConfig
from pitchvision.detection import Detection
from pitchvision.team_clustering import assign_team_clusters, _enhance_crop, TeamClusteringConfig


class QualityHeuristicsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Construct pipeline config with low min_crop_area so test bboxes are processed
        self.config = PipelineConfig(
            team_clustering=TeamClusteringConfig(
                min_crop_area=50,
                outlier_threshold=50.0,
                contrast_clip_limit=3.0,
                saturation_boost=1.8,
                value_boost=1.1,
                use_predefined_colors=False
            )
        )
        # Create a dummy image
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)
        # Green background
        self.image[:, :] = [42, 118, 70]

    def test_enhance_crop_modifies_colors(self) -> None:
        # Create a crop with low contrast/saturation
        crop = np.full((20, 20, 3), 120, dtype=np.uint8)
        # Add some color
        crop[5:15, 5:15] = [140, 110, 100]
        
        tc_config = TeamClusteringConfig()
        enhanced = _enhance_crop(crop, tc_config)
        
        self.assertEqual(enhanced.shape, crop.shape)
        # The pixel values should change due to CLAHE and Saturation boost
        self.assertFalse(np.array_equal(enhanced, crop))

    def test_goalkeeper_assignment_by_spatial_coordinates(self) -> None:
        # Create a dummy green field mask (entire image is field)
        field_mask = np.full((100, 200), 255, dtype=np.uint8)
        
        # We place players:
        # 1. Left GK (x = 5, y = 50 in image coordinates -> projected near left goal)
        # 2. Right GK (x = 195, y = 50 in image coordinates -> projected near right goal)
        # 3. Team A players (blue)
        # 4. Team B players (purple)
        # 5. Referee player (yellow)
        
        # Left GK: bright orange-red (20, 20, 255)
        self.image[40:60, 0:10] = [20, 20, 255]
        # Right GK: bright orange-red (20, 20, 255)
        self.image[40:60, 190:200] = [20, 20, 255]
        # Referee: bright yellow (50, 250, 250)
        self.image[40:60, 90:110] = [50, 250, 250]
        
        # Set Team A colors (blue)
        for x in (30, 40, 50, 60, 70):
            self.image[40:60, x:x+15] = [255, 50, 50]
            
        # Set Team B colors (purple)
        for x in (120, 130, 140, 150, 160):
            self.image[40:60, x:x+15] = [150, 50, 150]
        
        detections = [
            Detection(0, 0, "person", 0.9, (0, 40, 10, 60)),     # Left GK
            Detection(1, 0, "person", 0.9, (190, 40, 200, 60)), # Right GK
            Detection(2, 0, "person", 0.9, (50, 40, 65, 60)),   # Team A player
            Detection(3, 0, "person", 0.9, (130, 40, 145, 60)), # Team B player
            Detection(4, 0, "person", 0.9, (90, 40, 105, 60)),  # Referee
        ]
        
        # Add additional outfield players to dominate the K-means clustering
        for idx, x in enumerate((30, 40, 60, 70), start=5):
            detections.append(Detection(idx, 0, "person", 0.9, (x, 40, x+15, 60)))
        for idx, x in enumerate((120, 140, 150, 160), start=9):
            detections.append(Detection(idx, 0, "person", 0.9, (x, 40, x+15, 60)))
        
        clusters = assign_team_clusters(
            self.image,
            detections,
            self.config.team_clustering,
            field_mask=field_mask,
            visible_width_m=105.0
        )
        
        # Verify roles are assigned
        labels = [d.team_label for d in detections]
        self.assertIn("Goalkeeper A", labels)
        self.assertIn("Goalkeeper B", labels)
        self.assertIn("Referee", labels)
        self.assertIn("Team A", labels)
        self.assertIn("Team B", labels)

    def test_enforces_team_size_cap(self) -> None:
        field_mask = np.full((100, 200), 255, dtype=np.uint8)
        
        # Create 15 players belonging to Team A (same blue color)
        detections = []
        for i in range(15):
            x = 10 + i * 10
            self.image[40:60, x:x+8] = [255, 50, 50]
            detections.append(Detection(i, 0, "person", 0.9, (x, 40, x+8, 60)))
            
        assign_team_clusters(
            self.image,
            detections,
            self.config.team_clustering,
            field_mask=field_mask
        )
        
        # Verify that at most 11 players have a team label (Team A)
        team_a_count = sum(1 for d in detections if d.team_label == "Team A")
        # 11 maximum total per team
        self.assertLessEqual(team_a_count, 11)

    def test_predefined_colors_classification(self) -> None:
        # Construct pipeline config with predefined colors enabled
        config = PipelineConfig(
            team_clustering=TeamClusteringConfig(
                min_crop_area=50,
                use_predefined_colors=True,
                team_a_color_rgb=(15, 30, 130), # Chelsea Blue
                team_b_color_rgb=(95, 35, 40),  # Burnley Claret
                referee_color_rgb=(55, 55, 55), # Referee Black
                goalkeeper_color_rgb=(90, 160, 50) # Goalkeeper Green
            )
        )
        
        field_mask = np.full((100, 200), 255, dtype=np.uint8)
        
        # 1. Left GK: Green (BGR: [80, 160, 50] -> RGB: [50, 160, 80])
        self.image[40:60, 0:10] = [80, 160, 50]
        # 2. Chelsea Blue (BGR: [130, 30, 15] -> RGB: [15, 30, 130])
        self.image[40:60, 30:40] = [130, 30, 15]
        # 3. Burnley Claret (BGR: [40, 35, 95] -> RGB: [95, 35, 40])
        self.image[40:60, 70:80] = [40, 35, 95]
        # 4. Referee Black (BGR: [30, 30, 30] -> RGB: [30, 30, 30])
        self.image[40:60, 110:120] = [30, 30, 30]
        
        detections = [
            Detection(0, 0, "person", 0.9, (0, 40, 10, 60)),     # Left GK
            Detection(1, 0, "person", 0.9, (30, 40, 40, 60)),    # Chelsea Player
            Detection(2, 0, "person", 0.9, (70, 40, 80, 60)),    # Burnley Player
            Detection(3, 0, "person", 0.9, (110, 40, 120, 60)),  # Referee
        ]
        
        assign_team_clusters(
            self.image,
            detections,
            config.team_clustering,
            field_mask=field_mask,
            visible_width_m=105.0
        )
        
        # Verify labels match predefined roles
        self.assertEqual(detections[0].team_label, "Goalkeeper A")
        self.assertEqual(detections[1].team_label, "Team A")
        self.assertEqual(detections[2].team_label, "Team B")
        self.assertEqual(detections[3].team_label, "Referee")

    def test_referee_validation_distance_threshold(self) -> None:
        # Construct pipeline config with predefined colors enabled
        config = PipelineConfig(
            team_clustering=TeamClusteringConfig(
                min_crop_area=50,
                use_predefined_colors=True,
                team_a_color_rgb=(15, 30, 130), # Chelsea Blue
                team_b_color_rgb=(95, 35, 40),  # Burnley Claret
                referee_color_rgb=(55, 55, 55), # Referee Black
                goalkeeper_color_rgb=(90, 160, 50)
            )
        )
        
        field_mask = np.full((100, 200), 255, dtype=np.uint8)
        
        # Place a player with a gray kit that is far from referee color (RGB: [120, 120, 120])
        self.image[40:60, 110:120] = [120, 120, 120]
        
        detections = [
            Detection(0, 0, "person", 0.9, (110, 40, 120, 60)),
        ]
        
        assign_team_clusters(
            self.image,
            detections,
            config.team_clustering,
            field_mask=field_mask,
            visible_width_m=105.0
        )
        
        # Verify that they were not kept as Referee, but reclassified to Team B (closer to Burnley Claret than Chelsea Blue)
        self.assertNotEqual(detections[0].team_label, "Referee")
        self.assertEqual(detections[0].team_label, "Team B")


if __name__ == "__main__":
    unittest.main()
