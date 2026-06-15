from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .config import TeamClusteringConfig
from .detection import Detection
from .projection import MiniMapConfig, estimate_pitch_homography, project_point


@dataclass(frozen=True)
class TeamCluster:
    label: str
    color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class PlayerAppearance:
    detection: Detection
    feature: np.ndarray
    top_colors_rgb: list[tuple[int, int, int]]
    dominant_rgb: tuple[int, int, int] | None
    center_xy: tuple[int, int]


def _upper_body_crop(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return crop_bgr
    height, width = crop_bgr.shape[:2]
    top = crop_bgr[: max(1, int(height * 0.62)), :]
    margin = max(0, int(width * 0.12))
    if margin == 0 or width - margin <= margin:
        return top
    return top[:, margin : width - margin]


def _grass_mask(crop_bgr: np.ndarray, config: TeamClusteringConfig) -> np.ndarray:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    grass_low, grass_high = config.grass_hue_range
    return ((hue >= grass_low) & (hue <= grass_high) & (sat > 35) & (val > 35)).astype(np.uint8) * 255


def _enhance_crop(crop_bgr: np.ndarray, config: TeamClusteringConfig) -> np.ndarray:
    if crop_bgr.size == 0:
        return crop_bgr
    # 1. CLAHE in LAB space
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.contrast_clip_limit, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 2. Boost saturation and value in HSV space
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * config.saturation_boost, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.float32) * config.value_boost, 0, 255).astype(np.uint8)
    enhanced_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def _player_pixels(crop_bgr: np.ndarray, config: TeamClusteringConfig, is_cropped_and_enhanced: bool = False) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    crop = crop_bgr if is_cropped_and_enhanced else _enhance_crop(_upper_body_crop(crop_bgr), config)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass_mask = _grass_mask(crop, config)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    useful_mask = (grass_mask == 0) & (sat > 25) & (val > 35)
    pixels = crop[useful_mask]
    if len(pixels) < 40:
        pixels = crop.reshape(-1, 3)
    return pixels


def _top_colors(crop_bgr: np.ndarray, config: TeamClusteringConfig, max_colors: int = 3, is_cropped_and_enhanced: bool = False) -> list[tuple[int, int, int]]:
    pixels = _player_pixels(crop_bgr, config, is_cropped_and_enhanced=is_cropped_and_enhanced)
    if len(pixels) == 0:
        return []

    sample = pixels
    if len(sample) > 3000:
        indices = np.linspace(0, len(sample) - 1, 3000, dtype=int)
        sample = sample[indices]

    clusters = min(max_colors, len(sample))
    if clusters <= 0:
        return []

    kmeans = KMeans(n_clusters=clusters, n_init=8, random_state=7)
    labels = kmeans.fit_predict(sample)
    colors: list[tuple[int, int, int]] = []
    for cluster_idx in np.argsort(np.bincount(labels))[::-1]:
        center_bgr = kmeans.cluster_centers_[int(cluster_idx)]
        center_rgb = center_bgr[::-1]
        colors.append(tuple(int(max(0, min(255, round(c)))) for c in center_rgb))
    return colors[:max_colors]


def _appearance_feature(crop_bgr: np.ndarray, config: TeamClusteringConfig) -> tuple[np.ndarray, list[tuple[int, int, int]], tuple[int, int, int] | None]:
    crop = _upper_body_crop(crop_bgr)
    if crop.size == 0:
        return np.zeros(16, dtype=np.float32), [], None

    crop = _enhance_crop(crop, config)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = _grass_mask(crop, config)
    player_mask = cv2.bitwise_not(mask)

    h_bins, s_bins, v_bins = config.histogram_bins
    hist_h = cv2.calcHist([hsv], [0], player_mask, [h_bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], player_mask, [s_bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], player_mask, [v_bins], [0, 256])

    feature = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()]).astype(np.float32)
    total = float(feature.sum())
    if total > 0:
        feature /= total

    colors = _top_colors(crop, config, max_colors=3, is_cropped_and_enhanced=True)
    dominant_rgb = colors[0] if colors else None
    if colors:
        feature = np.concatenate([feature, (np.array(colors, dtype=np.float32).flatten() / 255.0)]).astype(np.float32)
    return feature, colors, dominant_rgb


def _mean_color(color_list: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    if not color_list:
        return (128, 128, 128)
    arr = np.array(color_list, dtype=np.float32)
    avg = arr.mean(axis=0)
    return tuple(int(max(0, min(255, round(v)))) for v in avg)


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    color = np.uint8([[list(rgb)]]).reshape(1, 1, 3)
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def _color_distance(a: np.ndarray | tuple | list, b: np.ndarray | tuple | list) -> float:
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(arr_a - arr_b))


def _bbox_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _bbox_bottom_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) // 2, y2)


def _inside_field_mask(field_mask: np.ndarray | None, point_xy: tuple[int, int]) -> bool:
    if field_mask is None or field_mask.size == 0:
        return True
    x, y = point_xy
    if x < 0 or y < 0 or y >= field_mask.shape[0] or x >= field_mask.shape[1]:
        return False
    return field_mask[y, x] > 0


def _field_center_score(point_xy: tuple[int, int], field_rect: tuple[int, int, int, int] | None) -> float:
    if field_rect is None:
        return 1.0
    x, y = point_xy
    x1, y1, x2, y2 = field_rect
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    nx = abs(x - center_x) / (width / 2.0)
    ny = abs(y - center_y) / (height / 2.0)
    dist = min(1.0, float(np.sqrt(nx * nx + ny * ny)))
    return 1.0 - dist


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: int
    label: str
    count: int
    mean_saturation: float
    mean_field_score: float
    mean_color_rgb: tuple[int, int, int]


def _cluster_stats(
    players: list[PlayerAppearance],
    labels: np.ndarray,
    cluster_id: int,
    field_rect: tuple[int, int, int, int] | None,
) -> ClusterSummary:
    members = [idx for idx, value in enumerate(labels) if int(value) == cluster_id]
    mean_saturation = 0.0
    mean_field_score = 0.0
    colors: list[tuple[int, int, int]] = []

    for idx in members:
        player = players[idx]
        if player.dominant_rgb is not None:
            _, sat, _ = _rgb_to_hsv(player.dominant_rgb)
            mean_saturation += sat
            colors.append(player.dominant_rgb)
        mean_field_score += _field_center_score(_bbox_bottom_center(player.detection.bbox_xyxy), field_rect)

    count = len(members)
    if count > 0:
        mean_saturation /= count
        mean_field_score /= count

    return ClusterSummary(
        cluster_id=cluster_id,
        label="",
        count=count,
        mean_saturation=mean_saturation,
        mean_field_score=mean_field_score,
        mean_color_rgb=_mean_color(colors),
    )


def _choose_referee_cluster(
    players: list[PlayerAppearance],
    labels: np.ndarray,
    centers: np.ndarray,
    field_rect: tuple[int, int, int, int] | None,
) -> int | None:
    if len(centers) < 3:
        return None

    summaries = [
        _cluster_stats(players, labels, cluster_id, field_rect)
        for cluster_id in range(len(centers))
    ]

    if not summaries:
        return None

    max_referee_count = max(2, int(np.ceil(len(players) * 0.2)))

    scores: list[tuple[float, int, int]] = []
    for summary in summaries:
        if summary.count == 0 or summary.count > max_referee_count:
            continue
        compactness = 1.0 / (1.0 + summary.count)
        neutrality = 1.0 - min(1.0, summary.mean_saturation / 255.0)
        score = 0.50 * summary.mean_field_score + 0.30 * compactness + 0.20 * neutrality
        scores.append((score, summary.cluster_id, summary.count))

    if not scores:
        return None

    scores.sort(reverse=True)
    best_score, best_cluster, best_count = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    if best_score < 0.56:
        return None
    if best_count > max_referee_count:
        return None
    if best_score - second_score < 0.05:
        return None
    return best_cluster


def _team_centroids(players: list[PlayerAppearance]) -> dict[str, np.ndarray]:
    team_points: dict[str, list[np.ndarray]] = {"Team A": [], "Team B": []}
    for player in players:
        if player.detection.team_label in team_points:
            team_points[player.detection.team_label].append(np.array(player.center_xy, dtype=np.float32))

    centroids: dict[str, np.ndarray] = {}
    for team_name, points in team_points.items():
        if points:
            centroids[team_name] = np.mean(points, axis=0)
    return centroids


def _team_color_centroids(players: list[PlayerAppearance]) -> dict[str, np.ndarray]:
    team_points: dict[str, list[np.ndarray]] = {"Team A": [], "Team B": []}
    for player in players:
        if player.detection.team_label in team_points and player.dominant_rgb is not None:
            team_points[player.detection.team_label].append(np.array(player.dominant_rgb, dtype=np.float32))

    centroids: dict[str, np.ndarray] = {}
    for team_name, points in team_points.items():
        if points:
            centroids[team_name] = np.mean(points, axis=0)
    return centroids


def _assign_team_by_hybrid(
    player: PlayerAppearance,
    color_centroids: dict[str, np.ndarray],
    spatial_centroids: dict[str, np.ndarray],
    config: TeamClusteringConfig,
) -> str:
    if player.dominant_rgb is None or len(color_centroids) < 2:
        return "Team A"

    color = np.array(player.dominant_rgb, dtype=np.float32)
    team_a_color = color_centroids["Team A"]
    team_b_color = color_centroids["Team B"]
    color_a = _color_distance(color, team_a_color)
    color_b = _color_distance(color, team_b_color)

    # Goalkeeper-like outliers often have kit colors far from both teams.
    goalkeeper_like = min(color_a, color_b) > 38.0 * config.goalkeeper_distance_factor
    if goalkeeper_like and len(spatial_centroids) == 2:
        point = np.array(player.center_xy, dtype=np.float32)
        spatial_a = float(np.linalg.norm(point - spatial_centroids["Team A"]))
        spatial_b = float(np.linalg.norm(point - spatial_centroids["Team B"]))
        return "Team A" if spatial_a <= spatial_b else "Team B"

    return "Team A" if color_a <= color_b else "Team B"


def assign_team_clusters(
    image_bgr: np.ndarray,
    detections: list[Detection],
    config: TeamClusteringConfig,
    field_rect: tuple[int, int, int, int] | None = None,
    field_mask: np.ndarray | None = None,
    camera_x_center: float = 52.5,
    visible_width_m: float = 48.0,
) -> list[TeamCluster]:
    height, width = image_bgr.shape[:2]
    players: list[PlayerAppearance] = []

    for detection in detections:
        if detection.class_name != "person":
            continue

        x1, y1, x2, y2 = detection.bbox_xyxy
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < config.min_crop_area:
            continue

        center = _bbox_bottom_center((x1, y1, x2, y2))
        if not _inside_field_mask(field_mask, center):
            continue
        if field_rect is not None:
            fx1, fy1, fx2, fy2 = field_rect
            if not (fx1 <= center[0] <= fx2 and fy1 <= center[1] <= fy2):
                continue

        crop = image_bgr[y1:y2, x1:x2]
        feature, colors, dominant_rgb = _appearance_feature(crop, config)
        if feature.size == 0:
            continue

        players.append(
            PlayerAppearance(
                detection=detection,
                feature=feature,
                top_colors_rgb=colors,
                dominant_rgb=dominant_rgb,
                center_xy=center,
            )
        )

    if not players:
        return []

    # Estimate temporary homography to get projected coordinates on the pitch for spatial heuristics
    minimap_config = MiniMapConfig()
    homography = estimate_pitch_homography(
        field_mask,
        image_bgr.shape,
        minimap_config,
        camera_x_center=camera_x_center,
        visible_width_m=visible_width_m,
    )
    
    # 2. Assign team and role labels
    # Calculate bounds for pixel-to-meter conversion
    left_px = minimap_config.margin_px
    right_px = minimap_config.width_px - minimap_config.margin_px
    
    def to_m(px_val: float) -> float:
        return (px_val - left_px) / (right_px - left_px) * 105.0

    # Project and set coordinates first so we have them for spatial rules
    for player in players:
        px, py = project_point(player.center_xy, homography)
        player.detection.pitch_xy = (px, py)
        player.detection.dominant_color_rgb = player.dominant_rgb
        player.detection.dominant_colors_rgb = player.top_colors_rgb

    if config.use_predefined_colors:
        # Zero-Shot Nearest Prototype classification using HSV rules and RGB fallback
        for player in players:
            if player.dominant_rgb is None:
                player.detection.team_label = "Team A"
                continue
            
            rgb = player.dominant_rgb
            h, s, v = _rgb_to_hsv(rgb)
            
            # 1. Green Goalkeeper (typically green jersey)
            if 35 <= h <= 85 and s >= 45 and v >= 45:
                player.detection.team_label = "Goalkeeper"
            # 2. Black Referee: must be dark (V < 90) and low-ish saturation (S < 95)
            elif v < 90 and s < 95 and (v < 50 or (s < 85 and v < 80)):
                player.detection.team_label = "Referee"
            # 3. Chelsea Blue (Team A): blue hue
            elif 95 <= h <= 135 and s >= 50 and v >= 40:
                player.detection.team_label = "Team A"
            # 4. Burnley Claret (Team B): red/purple hue
            elif (h < 25 or h > 145) and s >= 50 and v >= 40:
                player.detection.team_label = "Team B"
            # 5. Fallback: match by Euclidean RGB distance to Team A (Blue), Team B (Claret), or Referee (Black) prototypes
            else:
                dist_a = _color_distance(rgb, config.team_a_color_rgb)
                dist_b = _color_distance(rgb, config.team_b_color_rgb)
                dist_ref = _color_distance(rgb, config.referee_color_rgb)
                
                best_dist = min(dist_a, dist_b, dist_ref)
                if best_dist == dist_a:
                    player.detection.team_label = "Team A"
                elif best_dist == dist_b:
                    player.detection.team_label = "Team B"
                else:
                    player.detection.team_label = "Referee"
        
        # Determine team defending direction based on players classified as Team A vs Team B
        team_a_xs = [to_m(p.detection.pitch_xy[0]) for p in players if p.detection.team_label == "Team A"]
        team_b_xs = [to_m(p.detection.pitch_xy[0]) for p in players if p.detection.team_label == "Team B"]
        
        mean_x_a = np.mean(team_a_xs) if team_a_xs else 25.0
        mean_x_b = np.mean(team_b_xs) if team_b_xs else 75.0
        
        left_team = "Team A" if mean_x_a <= mean_x_b else "Team B"
        right_team = "Team B" if mean_x_a <= mean_x_b else "Team A"
        
        # Enforce Goalkeeper cap (max 1 per side)
        gka_candidates = []
        gkb_candidates = []
        for player in players:
            det = player.detection
            mx = to_m(det.pitch_xy[0])
            is_gk_color = (det.team_label == "Goalkeeper")
            is_gk_spatial = (det.team_label in ("Team A", "Team B") and (mx < 6.0 or mx > 99.0))
            if is_gk_color or is_gk_spatial:
                if mx < 52.5:
                    gka_candidates.append((mx, player))
                else:
                    gkb_candidates.append((105.0 - mx, player))
                    
        if gka_candidates:
            gka_candidates.sort(key=lambda item: item[0])
            gka_candidates[0][1].detection.team_label = "Goalkeeper A" if left_team == "Team A" else "Goalkeeper B"
            for _, p in gka_candidates[1:]:
                p.detection.team_label = left_team
                
        if gkb_candidates:
            gkb_candidates.sort(key=lambda item: item[0])
            gkb_candidates[0][1].detection.team_label = "Goalkeeper B" if right_team == "Team B" else "Goalkeeper A"
            for _, p in gkb_candidates[1:]:
                p.detection.team_label = right_team
                
        # Enforce Referee cap (max 1) and validation
        ref_candidates = []
        for player in players:
            det = player.detection
            if det.team_label == "Referee":
                if player.dominant_rgb is not None:
                    dist_ref = _color_distance(player.dominant_rgb, config.referee_color_rgb)
                    if dist_ref < 55.0:
                        ref_candidates.append((dist_ref, player))
                    else:
                        rgb = player.dominant_rgb
                        dist_a = _color_distance(rgb, config.team_a_color_rgb)
                        dist_b = _color_distance(rgb, config.team_b_color_rgb)
                        det.team_label = "Team A" if dist_a < dist_b else "Team B"
                else:
                    det.team_label = "Team A"
                    
        if ref_candidates:
            ref_candidates.sort(key=lambda item: item[0])
            ref_candidates[0][1].detection.team_label = "Referee"
            for _, p in ref_candidates[1:]:
                rgb = p.dominant_rgb or (128, 128, 128)
                dist_a = _color_distance(rgb, config.team_a_color_rgb)
                dist_b = _color_distance(rgb, config.team_b_color_rgb)
                p.detection.team_label = "Team A" if dist_a < dist_b else "Team B"
                
    else:
        # Fallback to K-Means logic (useful for backwards compatibility / tests)
        num_players = len(players)
        cluster_count = min(2, num_players)
        features = np.array([player.feature for player in players], dtype=np.float32)
        
        if cluster_count == 1:
            labels = np.zeros(num_players, dtype=int)
        else:
            kmeans = KMeans(n_clusters=cluster_count, n_init=20, random_state=7)
            labels = kmeans.fit_predict(features)
            
        temp_centroids = {}
        for c_id in range(cluster_count):
            member_colors = [players[idx].dominant_rgb for idx, l in enumerate(labels) if l == c_id and players[idx].dominant_rgb is not None]
            if member_colors:
                temp_centroids[c_id] = np.mean(member_colors, axis=0)
            else:
                temp_centroids[c_id] = np.array([128.0, 128.0, 128.0])
                
        cluster_to_team = {}
        if cluster_count == 2:
            hue_0, _, _ = _rgb_to_hsv(tuple(int(c) for c in temp_centroids[0]))
            hue_1, _, _ = _rgb_to_hsv(tuple(int(c) for c in temp_centroids[1]))
            if hue_0 <= hue_1:
                cluster_to_team[0] = "Team A"
                cluster_to_team[1] = "Team B"
            else:
                cluster_to_team[0] = "Team B"
                cluster_to_team[1] = "Team A"
        else:
            cluster_to_team[0] = "Team A"
            
        for idx, player in enumerate(players):
            player.detection.team_label = cluster_to_team[labels[idx]]
            
        centroid_A = np.array([128.0, 128.0, 128.0])
        centroid_B = np.array([128.0, 128.0, 128.0])
        for c_id, team_name in cluster_to_team.items():
            if team_name == "Team A":
                centroid_A = temp_centroids[c_id]
            elif team_name == "Team B":
                centroid_B = temp_centroids[c_id]
        
        clean_a = []
        clean_b = []
        for player in players:
            if player.dominant_rgb is None:
                continue
            color = np.array(player.dominant_rgb, dtype=np.float32)
            dist_a = _color_distance(color, centroid_A)
            dist_b = _color_distance(color, centroid_B)
            if dist_a < config.outlier_threshold:
                clean_a.append(color)
            if dist_b < config.outlier_threshold:
                clean_b.append(color)
                
        if clean_a:
            centroid_A = np.mean(clean_a, axis=0)
        if clean_b:
            centroid_B = np.mean(clean_b, axis=0)
            
        player_pitch_xs = [player.detection.pitch_xy[0] for player in players]
        team_a_xs = [player_pitch_xs[idx] for idx, p in enumerate(players) if p.detection.team_label == "Team A"]
        team_b_xs = [player_pitch_xs[idx] for idx, p in enumerate(players) if p.detection.team_label == "Team B"]
        
        mean_x_a = np.mean(team_a_xs) if team_a_xs else 25.0
        mean_x_b = np.mean(team_b_xs) if team_b_xs else 75.0
        
        left_team = "Team A" if mean_x_a <= mean_x_b else "Team B"
        right_team = "Team B" if mean_x_a <= mean_x_b else "Team A"
        
        goalkeepers = []
        referees = []
        for player in players:
            detection = player.detection
            if player.dominant_rgb is None:
                detection.team_label = "Team A"
                continue
                
            color = np.array(player.dominant_rgb, dtype=np.float32)
            dist_a = _color_distance(color, centroid_A)
            dist_b = _color_distance(color, centroid_B)
            min_dist = min(dist_a, dist_b)
            
            mx = to_m(detection.pitch_xy[0])
            is_color_outlier = min_dist > config.outlier_threshold
            is_spatial_gk_left = (mx < 12.0)
            is_spatial_gk_right = (mx > 93.0)
            
            if is_color_outlier or is_spatial_gk_left or is_spatial_gk_right:
                if mx < 22.0:
                    goalkeepers.append((mx, detection, left_team, "Goalkeeper A" if left_team == "Team A" else "Goalkeeper B"))
                elif mx > 83.0:
                    goalkeepers.append((105.0 - mx, detection, right_team, "Goalkeeper B" if right_team == "Team B" else "Goalkeeper A"))
                else:
                    midfield_dist = abs(mx - 52.5)
                    referees.append((midfield_dist, detection))
            else:
                detection.team_label = "Team A" if dist_a <= dist_b else "Team B"
                
        gks_left = [g for g in goalkeepers if to_m(g[1].pitch_xy[0]) < 52.5]
        gks_right = [g for g in goalkeepers if to_m(g[1].pitch_xy[0]) >= 52.5]
        if gks_left:
            gks_left.sort(key=lambda g: to_m(g[1].pitch_xy[0]))
            best_left_gk = gks_left[0]
            best_left_gk[1].team_label = "Goalkeeper A" if left_team == "Team A" else "Goalkeeper B"
            for g in gks_left[1:]:
                g[1].team_label = left_team
                
        if gks_right:
            gks_right.sort(key=lambda g: to_m(g[1].pitch_xy[0]), reverse=True)
            best_right_gk = gks_right[0]
            best_right_gk[1].team_label = "Goalkeeper B" if right_team == "Team B" else "Goalkeeper A"
            for g in gks_right[1:]:
                g[1].team_label = right_team
                
        referees.sort(key=lambda r: r[0])
        for dist, det in referees[:2]:
            det.team_label = "Referee"
        for dist, det in referees[2:]:
            color = np.array(det.dominant_color_rgb, dtype=np.float32)
            det.team_label = "Team A" if _color_distance(color, centroid_A) <= _color_distance(color, centroid_B) else "Team B"

    # Enforce team size cap (max 11 total per team including GK)
    for team_name in ("Team A", "Team B"):
        team_players = [p.detection for p in players if p.detection.team_label == team_name]
        gk_label = "Goalkeeper A" if team_name == "Team A" else "Goalkeeper B"
        team_gk = [p.detection for p in players if p.detection.team_label == gk_label]
        
        max_outfield = 11 - len(team_gk)
        if len(team_players) > max_outfield:
            team_players.sort(key=lambda det: det.confidence, reverse=True)
            overflow = team_players[max_outfield:]
            for det in overflow:
                det.team_label = None

    # 5. Build TeamCluster results
    clusters: list[TeamCluster] = []
    
    colors_a = [p.dominant_rgb for p in players if p.detection.team_label == "Team A" and p.dominant_rgb is not None]
    clusters.append(TeamCluster(label="Team A", color_rgb=_mean_color(colors_a) if colors_a else config.team_a_color_rgb))
    
    colors_b = [p.dominant_rgb for p in players if p.detection.team_label == "Team B" and p.dominant_rgb is not None]
    clusters.append(TeamCluster(label="Team B", color_rgb=_mean_color(colors_b) if colors_b else config.team_b_color_rgb))
    
    colors_gka = [p.dominant_rgb for p in players if p.detection.team_label == "Goalkeeper A" and p.dominant_rgb is not None]
    clusters.append(TeamCluster(label="Goalkeeper A", color_rgb=_mean_color(colors_gka) if colors_gka else (50, 220, 50)))
    
    colors_gkb = [p.dominant_rgb for p in players if p.detection.team_label == "Goalkeeper B" and p.dominant_rgb is not None]
    clusters.append(TeamCluster(label="Goalkeeper B", color_rgb=_mean_color(colors_gkb) if colors_gkb else (220, 50, 220)))

    colors_ref = [p.dominant_rgb for p in players if p.detection.team_label == "Referee" and p.dominant_rgb is not None]
    clusters.append(TeamCluster(label="Referee", color_rgb=_mean_color(colors_ref) if colors_ref else (0, 240, 240)))

    return clusters
