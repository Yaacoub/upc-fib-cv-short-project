from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML.
    yaml = None


@dataclass(frozen=True)
class TeamClusteringConfig:
    clusters: int = 3
    min_crop_area: int = 600
    grass_hue_range: tuple[int, int] = (35, 95)
    histogram_bins: tuple[int, int, int] = (8, 4, 4)
    goalkeeper_distance_factor: float = 1.35
    max_players_per_team: int = 11
    contrast_clip_limit: float = 3.0
    saturation_boost: float = 1.8
    value_boost: float = 1.1
    outlier_threshold: float = 50.0
    use_predefined_colors: bool = True
    team_a_color_rgb: tuple[int, int, int] = (15, 30, 130)
    team_b_color_rgb: tuple[int, int, int] = (95, 35, 40)
    referee_color_rgb: tuple[int, int, int] = (55, 55, 55)
    goalkeeper_color_rgb: tuple[int, int, int] = (90, 160, 50)


@dataclass(frozen=True)
class FieldLineConfig:
    canny_low: int = 70
    canny_high: int = 170
    hough_threshold: int = 90
    min_line_length: int = 80
    max_line_gap: int = 12


@dataclass(frozen=True)
class DetectionFilterConfig:
    min_box_area_ratio: float = 0.0005
    max_box_area_ratio: float = 0.15
    min_box_width: int = 12
    min_box_height: int = 16
    person_min_confidence: float = 0.20
    ball_min_confidence: float = 0.01
    ball_min_box_area_ratio: float = 0.00001
    ball_min_box_width: int = 4
    ball_min_box_height: int = 4


@dataclass(frozen=True)
class ProjectionConfig:
    visible_pitch_width_m: float = 48.0


@dataclass
class PipelineConfig:
    model: str = "yolov8n.pt"
    confidence: float = 0.02
    iou: float = 0.45
    person_class: int = 0
    ball_class: int = 32
    team_clustering: TeamClusteringConfig = TeamClusteringConfig()
    field_lines: FieldLineConfig = FieldLineConfig()
    projection: ProjectionConfig = ProjectionConfig()
    detection_filter: DetectionFilterConfig = DetectionFilterConfig()


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML config files.")
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw)

    classes = data.get("classes", {})
    team = data.get("team_clustering", {})
    field = data.get("field_lines", {})
    proj = data.get("projection", {})

    return PipelineConfig(
        model=data.get("model", "yolov8n.pt"),
        confidence=float(data.get("confidence", 0.05)),
        iou=float(data.get("iou", 0.45)),
        person_class=int(classes.get("person", 0)),
        ball_class=int(classes.get("sports_ball", 32)),
        team_clustering=TeamClusteringConfig(
            clusters=int(team.get("clusters", 3)),
            min_crop_area=int(team.get("min_crop_area", 600)),
            grass_hue_range=tuple(team.get("grass_hue_range", [35, 95])),
            histogram_bins=tuple(team.get("histogram_bins", [8, 4, 4])),
            goalkeeper_distance_factor=float(team.get("goalkeeper_distance_factor", 1.35)),
            max_players_per_team=int(team.get("max_players_per_team", 11)),
            contrast_clip_limit=float(team.get("contrast_clip_limit", 3.0)),
            saturation_boost=float(team.get("saturation_boost", 1.8)),
            value_boost=float(team.get("value_boost", 1.1)),
            outlier_threshold=float(team.get("outlier_threshold", 50.0)),
            use_predefined_colors=bool(team.get("use_predefined_colors", True)),
            team_a_color_rgb=tuple(team.get("team_a_color_rgb", [15, 30, 130])),
            team_b_color_rgb=tuple(team.get("team_b_color_rgb", [95, 35, 40])),
            referee_color_rgb=tuple(team.get("referee_color_rgb", [55, 55, 55])),
            goalkeeper_color_rgb=tuple(team.get("goalkeeper_color_rgb", [90, 160, 50])),
        ),
        projection=ProjectionConfig(
            visible_pitch_width_m=float(proj.get("visible_pitch_width_m", 48.0)),
        ),
        field_lines=FieldLineConfig(
            canny_low=int(field.get("canny_low", 70)),
            canny_high=int(field.get("canny_high", 170)),
            hough_threshold=int(field.get("hough_threshold", 90)),
            min_line_length=int(field.get("min_line_length", 80)),
            max_line_gap=int(field.get("max_line_gap", 12)),
        ),
        detection_filter=DetectionFilterConfig(
            min_box_area_ratio=float(data.get("min_box_area_ratio", 0.0005)),
            max_box_area_ratio=float(data.get("max_box_area_ratio", 0.15)),
            min_box_width=int(data.get("min_box_width", 12)),
            min_box_height=int(data.get("min_box_height", 16)),
            person_min_confidence=float(data.get("person_min_confidence", 0.20)),
            ball_min_confidence=float(data.get("ball_min_confidence", 0.01)),
            ball_min_box_area_ratio=float(data.get("ball_min_box_area_ratio", 0.00001)),
            ball_min_box_width=int(data.get("ball_min_box_width", 4)),
            ball_min_box_height=int(data.get("ball_min_box_height", 4)),
        ),
    )
