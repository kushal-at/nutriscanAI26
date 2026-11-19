# -*- coding: utf-8 -*-
"""
vision/segmentation.py
======================
DeepLabV3+ semantic segmentation inference with aspect-ratio-preserving letterbox
padding, bilinear probability smoothing, and planar metrology scale calibration
using dual OpenCV Hough circle and ellipse rim fitting.
"""

import json
import os
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp


def load_segmenter(model_path: Path, labels_path: Path) -> Tuple[torch.nn.Module, Dict[int, str], torch.device]:
    """Load PyTorch DeepLabV3+ MobileNetV2 model and 104-class label dictionary."""
    if not model_path.is_file():
        raise RuntimeError(f"FoodSeg103 weights not found: {model_path}")
    if not labels_path.is_file():
        raise RuntimeError(f"FoodSeg103 class labels not found: {labels_path}")

    with labels_path.open(encoding="utf-8") as f:
        _label_config = json.load(f)
    class_names = {int(key): value for key, value in _label_config["id2label"].items()}
    if len(class_names) != 104:
        raise RuntimeError(f"Expected 104 FoodSeg103 classes, found {len(class_names)}")

    segmenter = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=104
    )
    segmenter.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmenter.to(device).eval()
    return segmenter, class_names, device


def segmentation_summary_letterbox(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    target_size: int = 512,
    confidence_thresh: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Strategy 3 (Production): Aspect-Ratio-Preserving Letterbox Padding to 512x512
    with Bilinear Probability Logit Upsampling.

    1. Resizes input image so its longest dimension is target_size (512), preserving aspect ratio.
    2. Places the scaled image onto a 512x512 canvas with neutral padding.
    3. Executes a single forward pass on DeepLabV3+ MobileNetV2.
    4. Removes padding in probability tensor space.
    5. Bilinearly interpolates continuous class probabilities to original resolution (W, H),
       preventing coarse nearest-neighbor blocky mask artifacts.
    """
    orig_w, orig_h = image.size

    # 1. Scale preserving aspect ratio so max dimension is target_size (512)
    scale = target_size / max(orig_w, orig_h)
    new_w = max(1, min(target_size, int(round(orig_w * scale))))
    new_h = max(1, min(target_size, int(round(orig_h * scale))))

    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 2. Place on 512x512 canvas with neutral ImageNet-mean padding (prevents edge contrast cliffs)
    canvas = Image.new("RGB", (target_size, target_size), (124, 116, 104))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized_img, (pad_x, pad_y))

    # 3. ImageNet normalization
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    normalized = (pixels - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # 4. DeepLabV3+ inference
    with torch.inference_mode():
        logits = segmenter(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]  # shape: (104, target_size, target_size)

    # 5. Crop out padding: isolate active food photo region in tensor space
    unpad_probs = probabilities[:, pad_y : pad_y + new_h, pad_x : pad_x + new_w]  # (104, new_h, new_w)

    # 6. Memory-Bounded Smooth Bilinear Resampling of continuous candidate probabilities
    # Filters down to candidate classes (prob > 0.04) and caps interpolation grid to 1280px
    # to eliminate 20GB+ OOM allocation bombs on 12-50MP photos while retaining smooth sub-pixel boundaries
    class_max_probs = unpad_probs.amax(dim=(1, 2))
    candidate_indices = torch.where(class_max_probs > 0.04)[0]
    if len(candidate_indices) == 0:
        candidate_indices = torch.tensor([0], device=unpad_probs.device)
    if 0 not in candidate_indices:
        candidate_indices = torch.cat([torch.tensor([0], device=unpad_probs.device), candidate_indices])

    cand_probs = unpad_probs[candidate_indices].unsqueeze(0)  # shape: (1, K, new_h, new_w) where K <= 10

    max_dim = max(orig_h, orig_w)
    if max_dim > 1280:
        scale_sub = 1280.0 / max_dim
        interp_h = max(1, int(round(orig_h * scale_sub)))
        interp_w = max(1, int(round(orig_w * scale_sub)))
    else:
        interp_h, interp_w = orig_h, orig_w

    sub_probs = torch.nn.functional.interpolate(
        cand_probs,
        size=(interp_h, interp_w),
        mode="bilinear",
        align_corners=False,
    )[0]  # shape: (K, interp_h, interp_w)

    conf_sub_t, local_pred_sub_t = sub_probs.max(dim=0)
    pred_sub_t = candidate_indices[local_pred_sub_t]

    pred_sub_np = pred_sub_t.cpu().numpy().astype(np.uint8)
    conf_sub_np = conf_sub_t.cpu().numpy().astype(np.float32)

    if (interp_h, interp_w) != (orig_h, orig_w):
        pred_orig = cv2.resize(pred_sub_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        conf_orig = cv2.resize(conf_sub_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        pred_orig = pred_sub_np
        conf_orig = conf_sub_np

    # 7. Extract class areas, confidences, and total food pixels
    areas: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    total_pixels = orig_w * orig_h
    food_pixels = 0

    # Effective confidence threshold (default 0.35 for fine boundary coverage without noise)
    eff_conf = max(0.20, min(confidence_thresh, 0.55))

    for class_id in np.unique(pred_orig):
        class_id = int(class_id)
        if class_id == 0 or class_id not in class_names:
            continue
        mask = pred_orig == class_id
        count = int(mask.sum())
        fraction = count / total_pixels
        mean_conf = float(conf_orig[mask].mean()) if count > 0 else 0.0
        if fraction < 0.0008 or mean_conf < eff_conf:
            continue
        class_name = class_names[class_id]
        food_pixels += count
        areas.append({"class": class_name, "area_percent": round(fraction * 100, 2)})
        detections.append({"class": class_name, "confidence": round(mean_conf, 4)})

    areas.sort(key=lambda item: item["area_percent"], reverse=True)
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return areas, detections, food_pixels, pred_orig, conf_orig


def segmentation_summary_tiled(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    tile_size: int = 512,
    overlap: int = 64,
    confidence_thresh: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Segment original-resolution image pixels using overlapping tiles (Rollback Strategy)."""
    width, height = image.size
    pixels = np.asarray(image, dtype=np.uint8)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def tile_starts(length: int) -> List[int]:
        if length <= tile_size:
            return [0]
        stride = tile_size - overlap
        starts = list(range(0, length - tile_size + 1, stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def blend_weights(start: int, valid_length: int, full_length: int) -> np.ndarray:
        weights = np.ones(valid_length, dtype=np.float32)
        ramp_length = min(overlap, valid_length)
        if ramp_length and start > 0:
            # Smooth half-cosine ramp from 0.40 to 1.0 (never penalize border foods down to 0.05)
            t = np.linspace(0, np.pi / 2, ramp_length, dtype=np.float32)
            weights[:ramp_length] = 0.40 + 0.60 * np.sin(t)
        if ramp_length and start + valid_length < full_length:
            t = np.linspace(0, np.pi / 2, ramp_length, dtype=np.float32)
            weights[-ramp_length:] = np.minimum(
                weights[-ramp_length:],
                0.40 + 0.60 * np.cos(t),
            )
        return weights

    pred = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    best_score = np.full((height, width), -1.0, dtype=np.float32)

    for y in tile_starts(height):
        valid_height = min(tile_size, height - y)
        input_height = max(32, ((valid_height + 31) // 32) * 32)
        y_weights = blend_weights(y, valid_height, height)

        for x in tile_starts(width):
            valid_width = min(tile_size, width - x)
            input_width = max(32, ((valid_width + 31) // 32) * 32)
            x_weights = blend_weights(x, valid_width, width)

            tile = pixels[y : y + valid_height, x : x + valid_width].astype(np.float32) / 255.0
            normalized = (tile - mean) / std
            tensor_array = np.zeros((1, 3, input_height, input_width), dtype=np.float32)
            tensor_array[0, :, :valid_height, :valid_width] = normalized.transpose(2, 0, 1)
            tensor = torch.from_numpy(tensor_array).to(device)

            with torch.inference_mode():
                logits = segmenter(tensor)
                probabilities = torch.softmax(logits, dim=1)[0, :, :valid_height, :valid_width]
                confidence, prediction = probabilities.max(dim=0)

            tile_pred = prediction.cpu().numpy().astype(np.uint8)
            tile_conf = confidence.cpu().numpy().astype(np.float32)
            tile_weight = y_weights[:, None] * x_weights[None, :]
            score = tile_conf * tile_weight
            score_region = best_score[y : y + valid_height, x : x + valid_width]
            replace = score > score_region
            pred_region = pred[y : y + valid_height, x : x + valid_width]
            confidence_region = confidence_map[y : y + valid_height, x : x + valid_width]
            pred_region[replace] = tile_pred[replace]
            confidence_region[replace] = tile_conf[replace]
            score_region[replace] = score[replace]

    areas: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    total_pixels = width * height
    food_pixels = 0
    for class_id in np.unique(pred):
        class_id = int(class_id)
        if class_id == 0 or class_id not in class_names:
            continue
        mask = pred == class_id
        count = int(mask.sum())
        fraction = count / total_pixels
        mean_conf = float(confidence_map[mask].mean())
        if fraction < 0.001 or mean_conf < confidence_thresh:
            continue
        class_name = class_names[class_id]
        food_pixels += count
        areas.append({"class": class_name, "area_percent": round(fraction * 100, 2)})
        detections.append({"class": class_name, "confidence": round(mean_conf, 4)})
    areas.sort(key=lambda item: item["area_percent"], reverse=True)
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return areas, detections, food_pixels


def segmentation_summary(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    tile_size: int = 512,
    overlap: int = 64,
    confidence_thresh: float = 0.25,
    strategy: str = "letterbox",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Run 104-class semantic segmentation on DeepLabV3+ MobileNetV2.

    Strategies:
      - 'letterbox' (Default, Phase 2): Aspect-ratio-preserving resize with neutral padding
        and bilinear continuous logit upsampling.
      - 'tiling' (Rollback Baseline): Overlapping 512x512 tiles with boundary blending.
    """
    selected_strategy = os.getenv("FOODSEG_STRATEGY", strategy).lower().strip()
    if selected_strategy in ("tiling", "tiles", "overlap"):
        return segmentation_summary_tiled(
            image, segmenter, device, class_names,
            tile_size=tile_size, overlap=overlap, confidence_thresh=confidence_thresh,
        )
    res = segmentation_summary_letterbox(
        image, segmenter, device, class_names,
        target_size=tile_size, confidence_thresh=confidence_thresh,
    )
    return res[0], res[1], res[2]


def extract_spatial_clusters(
    pred_orig: np.ndarray,
    conf_orig: np.ndarray,
    class_names: Dict[int, str],
    min_area_fraction: float = 0.003,
) -> List[Dict[str, Any]]:
    """
    Extract connected spatial component clusters from dense segmentation mask with
    normalized bounding boxes [y_min, x_min, y_max, x_max], centroids (x, y),
    and Ramer-Douglas-Peucker polygon vertices for SVG rendering.
    """
    h, w = pred_orig.shape[:2]
    total_pixels = h * w
    food_mask = (pred_orig > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(food_mask, connectivity=8)
    clusters: List[Dict[str, Any]] = []

    for i in range(1, num_labels):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        area_frac = area_px / total_pixels
        if area_frac < min_area_fraction:
            continue

        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        x1 = min(w, x0 + cw)
        y1 = min(h, y0 + ch)

        cluster_mask = (labels == i)
        cluster_classes = pred_orig[cluster_mask]
        cluster_confs = conf_orig[cluster_mask]

        vals, counts = np.unique(cluster_classes, return_counts=True)
        dominant_id = int(vals[np.argmax(counts)])
        dominant_name = class_names.get(dominant_id, "Food")
        mean_conf = float(cluster_confs.mean()) if len(cluster_confs) > 0 else 0.5

        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        # Extract contour polygon for SVG rendering
        comp_uint8 = cluster_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_pts: List[List[float]] = []
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            epsilon = max(1.5, 0.005 * cv2.arcLength(largest_cnt, True))
            approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
            polygon_pts = [[round(float(pt[0][0]) / w, 4), round(float(pt[0][1]) / h, 4)] for pt in approx]

        clusters.append({
            "cluster_id": len(clusters) + 1,
            "class": dominant_name,
            "label": dominant_name,
            "confidence": round(mean_conf, 3),
            "area_fraction": round(area_frac, 4),
            "area_percent": round(area_frac * 100, 2),
            "bbox_norm": [
                round(y0 / h, 4), round(x0 / w, 4),
                round(y1 / h, 4), round(x1 / w, 4),
            ],
            "bbox_px": [y0, x0, y1, x1],
            "centroid_norm": [round(cx / w, 4), round(cy / h, 4)],
            "centroid_px": [int(cx), int(cy)],
            "polygon_norm": polygon_pts,
        })

    clusters.sort(key=lambda c: c["area_fraction"], reverse=True)
    return clusters


def unrectify_polygon_points(
    polygon_norm: List[List[float]],
    rectified_dim: int,
    orig_w: int,
    orig_h: int,
    M_homography: Optional[np.ndarray],
) -> List[List[float]]:
    """Transform normalized polygon points from rectified square coordinates back into original image normalized space."""
    if not polygon_norm or M_homography is None:
        return polygon_norm
    try:
        M_inv = np.linalg.inv(M_homography)
        pts = np.array([[pt[0] * rectified_dim, pt[1] * rectified_dim] for pt in polygon_norm], dtype=np.float32).reshape(-1, 1, 2)
        unwarped = cv2.perspectiveTransform(pts, M_inv)
        return [[round(float(np.clip(p[0][0] / orig_w, 0.0, 1.0)), 4), round(float(np.clip(p[0][1] / orig_h, 0.0, 1.0)), 4)] for p in unwarped]
    except Exception:
        return polygon_norm


def rectify_perspective_plate(
    image: Image.Image,
    ellipse_info: Dict[str, Any],
) -> Tuple[Image.Image, Any, float]:
    """
    Rectify tilted handheld plate photo into a canonical circular top-view using homography.
    If plate is already circular (aspect_ratio >= 0.90), returns original image.
    """
    aspect = ellipse_info.get("aspect_ratio", 1.0)
    major_axis = ellipse_info.get("major_axis_px", 0.0)
    if aspect >= 0.90 or major_axis <= 10.0:
        return image, None, major_axis / 2.0

    img_np = np.asarray(image)
    h, w = img_np.shape[:2]
    cx = ellipse_info["cx"]
    cy = ellipse_info["cy"]
    major_r = ellipse_info["major_axis_px"] / 2.0
    minor_r = ellipse_info["minor_axis_px"] / 2.0
    angle_rad = np.deg2rad(ellipse_info.get("angle_deg", 0.0))

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    p1 = [cx - major_r * cos_a, cy - major_r * sin_a]
    p2 = [cx + major_r * cos_a, cy + major_r * sin_a]
    p3 = [cx - minor_r * -sin_a, cy - minor_r * cos_a]
    p4 = [cx + minor_r * -sin_a, cy + minor_r * cos_a]
    src_pts = np.array([p1, p2, p3, p4], dtype=np.float32)

    out_dim = int(np.clip(major_r * 2.1, 200, 2400))
    center_out = out_dim / 2.0
    d1 = [center_out - major_r, center_out]
    d2 = [center_out + major_r, center_out]
    d3 = [center_out, center_out - major_r]
    d4 = [center_out, center_out + major_r]
    dst_pts = np.array([d1, d2, d3, d4], dtype=np.float32)

    try:
        M_homography, _ = cv2.findHomography(src_pts, dst_pts)
        if M_homography is not None:
            rectified_np = cv2.warpPerspective(img_np, M_homography, (out_dim, out_dim), flags=cv2.INTER_LANCZOS4)
            return Image.fromarray(rectified_np), M_homography, major_r
    except Exception:
        pass

    return image, None, major_r


def segmentation_summary_with_clusters(
    image: Image.Image,
    segmenter: torch.nn.Module,
    device: torch.device,
    class_names: Dict[int, str],
    tile_size: int = 512,
    confidence_thresh: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """
    Run 104-class semantic segmentation and extract connected spatial component clusters
    with centroids, bounding boxes, and RDP contour polygons for SVG rendering and focal cropping.
    """
    areas, detections, food_pixels, pred_orig, conf_orig = segmentation_summary_letterbox(
        image, segmenter, device, class_names,
        target_size=tile_size, confidence_thresh=confidence_thresh,
    )
    clusters = extract_spatial_clusters(pred_orig, conf_orig, class_names)
    return areas, detections, food_pixels, clusters


def _detect_plate_hough(
    gray: np.ndarray,
    short_side: int,
    w: int,
    h: int,
    cx0: float,
    cy0: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Detect plate rim using OpenCV Hough circles."""
    gray_med = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray_med,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=short_side * 0.35,
        param1=100,
        param2=30,
        minRadius=int(short_side * 0.20),
        maxRadius=int(max(w, h) * 0.90),
    )
    if circles is None:
        return False, {}

    candidates = []
    for x, y, radius in circles[0]:
        # Allow plate to extend significantly outside image boundaries (up to 60% radius for cropped / close-up shots)
        margin = radius * 0.60
        if x - radius < -margin or y - radius < -margin or x + radius > w + margin or y + radius > h + margin:
            continue
        center_offset = float(np.hypot((x - cx0) / w, (y - cy0) / h))
        if center_offset > 0.65:
            continue
        candidates.append((center_offset, -radius, float(x), float(y), float(radius)))

    if not candidates:
        return False, {}

    candidates.sort()
    offset, _, cx, cy, radius = candidates[0]
    diameter_px = radius * 2.0
    return True, {
        "cx": cx,
        "cy": cy,
        "diameter_px": diameter_px,
        "radius": radius,
        "center_offset": offset,
    }


def _detect_plate_ellipse(
    gray: np.ndarray,
    short_side: int,
    w: int,
    h: int,
    cx0: float,
    cy0: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Detect plate rim using outer rim contour extraction and ellipse fitting (cv2.fitEllipse)."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Dual edge detection: Canny + adaptive threshold to detect plate boundaries under varied illumination
    v = float(np.median(blurred))
    lower = int(max(20, (1.0 - 0.35) * v))
    upper = int(min(220, (1.0 + 0.35) * v))
    edges_canny = cv2.Canny(blurred, lower, upper)

    adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    edges_adapt = cv2.Canny(adapt, 50, 150)
    combined = cv2.bitwise_or(edges_canny, edges_adapt)

    # Morphological closing with elliptical kernel to link broken plate rim segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_peri = short_side * 0.40
    candidates = []

    for cnt in contours:
        if len(cnt) < 10:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < min_peri:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except Exception:
            continue

        (cx, cy), (axis1, axis2), angle = ellipse
        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)

        if major_axis <= 0 or minor_axis <= 0:
            continue

        # Major axis can span 25% to 160% of frame dimensions (allows close-up & cropped plates)
        if not (short_side * 0.25 <= major_axis <= max(w, h) * 1.60):
            continue

        center_offset = float(np.hypot((cx - cx0) / w, (cy - cy0) / h))
        if center_offset > 0.65:
            continue

        aspect_ratio = minor_axis / major_axis
        # Support realistic handheld camera tilt angles (from 42 deg to flat 90 deg)
        if not (0.42 <= aspect_ratio <= 1.0):
            continue

        # Theoretical ellipse perimeter (Ramanujan first approximation)
        a = major_axis / 2.0
        b = minor_axis / 2.0
        h_param = ((a - b) ** 2) / ((a + b) ** 2 + 1e-6)
        theo_peri = np.pi * (a + b) * (1.0 + (3.0 * h_param) / (10.0 + np.sqrt(4.0 - 3.0 * h_param)))
        peri_ratio = min(peri, theo_peri) / max(peri, theo_peri)

        # Composite fitness score
        score = (peri_ratio * 0.40) + ((1.0 - center_offset) * 0.35) + ((major_axis / short_side) * 0.25)
        candidates.append((score, cx, cy, major_axis, minor_axis, angle, aspect_ratio, center_offset))

    if not candidates:
        return False, {}

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, cx, cy, major_axis, minor_axis, angle, aspect_ratio, center_offset = candidates[0]

    return True, {
        "cx": float(cx),
        "cy": float(cy),
        "major_axis_px": float(major_axis),
        "minor_axis_px": float(minor_axis),
        "angle_deg": float(angle),
        "aspect_ratio": float(aspect_ratio),
        "center_offset": float(center_offset),
        "score": float(best_score),
    }


def detect_top_plate_scale(image: Image.Image, plate_diameter_cm: float, food_pixels: int) -> Dict[str, Any]:
    """
    Calibrate top-view pixel scale from fully visible plate rim using dual Hough circle
    and robust outer contour ellipse fitting. Supports tilted/handheld camera angles.
    """
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    short_side = min(w, h)
    cx0, cy0 = w / 2.0, h / 2.0

    # 1. Run Hough Circle Detection
    hough_found, hough_res = _detect_plate_hough(gray, short_side, w, h, cx0, cy0)

    # 2. Run Ellipse Detection (robust to perspective foreshortening / angled shots)
    ellipse_found, ellipse_res = _detect_plate_ellipse(gray, short_side, w, h, cx0, cy0)

    if not hough_found and not ellipse_found:
        plate_physical_area_cm2 = np.pi * ((plate_diameter_cm / 2.0) ** 2)
        estimated_diameter_px = float(max(w, h) * 1.05)
        pixels_per_cm = estimated_diameter_px / plate_diameter_cm
        default_footprint = round(0.55 * plate_physical_area_cm2, 1)
        return {
            "detected": False,
            "confidence": "low",
            "reason": "plate rim partially visible or cropped; heuristic scale calibrated from photo bounds",
            "plate_diameter_px": round(estimated_diameter_px, 1),
            "minor_axis_px": round(estimated_diameter_px, 1),
            "cx": round(float(cx0), 1),
            "cy": round(float(cy0), 1),
            "aspect_ratio": 1.0,
            "tilt_angle_deg": 0.0,
            "pixels_per_cm": round(pixels_per_cm, 2),
            "food_footprint_cm2": default_footprint,
            "food_to_plate_ratio": 0.55,
            "method": f"Close-up / cropped photo heuristic scale ({plate_diameter_cm:g}cm nominal plate)",
            "ellipse_info": None,
        }

    # 3. Decision Logic between Circle and Ellipse Fitting
    use_ellipse = False
    if ellipse_found and not hough_found:
        use_ellipse = True
    elif ellipse_found and hough_found:
        # If ellipse reveals significant perspective tilt (aspect_ratio < 0.90) and has good score
        if ellipse_res["aspect_ratio"] < 0.90 and ellipse_res["score"] > 0.50:
            use_ellipse = True
        elif hough_res["center_offset"] > ellipse_res["center_offset"] + 0.10:
            use_ellipse = True

    if use_ellipse:
        major_axis_px = ellipse_res["major_axis_px"]
        minor_axis_px = ellipse_res["minor_axis_px"]
        aspect_ratio = ellipse_res["aspect_ratio"]
        tilt_angle_deg = float(np.degrees(np.arccos(np.clip(aspect_ratio, 0.0, 1.0))))
        center_offset = ellipse_res["center_offset"]

        # In perspective projection of a circle on a plane, the major axis is parallel to
        # the camera tilt axis and represents true un-foreshortened diameter
        diameter_px = major_axis_px
        pixels_per_cm = diameter_px / plate_diameter_cm
        plate_area_px = (np.pi / 4.0) * major_axis_px * minor_axis_px
        method = f"Fitted ellipse estimate (tilt-compensated {tilt_angle_deg:.1f}°)"
    else:
        radius = hough_res["radius"]
        diameter_px = radius * 2.0
        minor_axis_px = diameter_px
        aspect_ratio = 1.0
        tilt_angle_deg = 0.0
        center_offset = hough_res["center_offset"]

        pixels_per_cm = diameter_px / plate_diameter_cm
        plate_area_px = np.pi * radius * radius
        method = "Hough circle estimate from top-view plate rim"

    # 4. Realistic Coverage & Footprint Calculation
    # Note: Removed rigid self-destruct threshold (food_to_plate_ratio > 0.95);
    # a generously full plate of food should never abort plate calibration.
    food_to_plate_ratio = food_pixels / max(1.0, plate_area_px)
    coverage_ratio = min(food_to_plate_ratio, 1.0)
    plate_physical_area_cm2 = np.pi * ((plate_diameter_cm / 2.0) ** 2)
    footprint_cm2 = coverage_ratio * plate_physical_area_cm2

    # Confidence evaluation
    if center_offset < 0.15 and aspect_ratio >= 0.75:
        confidence = "high" if aspect_ratio >= 0.88 else "medium"
    elif center_offset < 0.28:
        confidence = "medium"
    else:
        confidence = "low"

    detected_cx = ellipse_res["cx"] if use_ellipse else hough_res["cx"]
    detected_cy = ellipse_res["cy"] if use_ellipse else hough_res["cy"]

    return {
        "detected": True,
        "confidence": confidence,
        "plate_diameter_px": round(diameter_px, 1),
        "minor_axis_px": round(minor_axis_px, 1),
        "cx": round(float(detected_cx), 1),
        "cy": round(float(detected_cy), 1),
        "aspect_ratio": round(aspect_ratio, 3),
        "tilt_angle_deg": round(tilt_angle_deg, 1),
        "pixels_per_cm": round(pixels_per_cm, 2),
        "food_footprint_cm2": round(footprint_cm2, 1),
        "food_to_plate_ratio": round(food_to_plate_ratio, 3),
        "method": method,
        "ellipse_info": ellipse_res if use_ellipse else None,
    }
