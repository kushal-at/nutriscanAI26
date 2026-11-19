# -*- coding: utf-8 -*-
"""
Food Vision API
===============
FastAPI service orchestrating dual-view food image analysis:
DeepLabV3+ semantic segmentation, tilt-compensated plate rim metrology,
and local Qwen3-VL VLM inference with unblocked ASGI event loop.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import anyio
import httpx
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

# Pillow Decompression Bomb Defense: Globally cap max pixels immediately after import
Image.MAX_IMAGE_PIXELS = 50_000_000

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

try:
    from vision.segmentation import (
        load_segmenter,
        segmentation_summary,
        segmentation_summary_with_clusters,
        detect_top_plate_scale,
        rectify_perspective_plate,
        extract_spatial_clusters,
        unrectify_polygon_points,
    )
    from vision.estimation import (
        density_category,
        build_weight_estimates,
        disaggregate_recipe_components,
    )
    from vision.vlm import (
        get_vlm_thumbnail_b64,
        extract_focal_crops,
        format_segmentation_priors,
        parse_component_response,
        query_qwen_vlm,
    )
except ModuleNotFoundError:
    from segmentation import (
        load_segmenter,
        segmentation_summary,
        segmentation_summary_with_clusters,
        detect_top_plate_scale,
        rectify_perspective_plate,
        extract_spatial_clusters,
        unrectify_polygon_points,
    )
    from estimation import (
        density_category,
        build_weight_estimates,
        disaggregate_recipe_components,
    )
    from vlm import (
        get_vlm_thumbnail_b64,
        extract_focal_crops,
        format_segmentation_priors,
        parse_component_response,
        query_qwen_vlm,
    )

SEGMENTER_PATH = Path(os.getenv("FOODSEG_MODEL", ROOT / "models/FoodSeg103/best_model.pth"))
LABELS_PATH = Path(os.getenv("FOODSEG_LABELS", ROOT / "models/FoodSeg103/labels_config.json"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl:2b")
QWEN_NUM_CTX = int(os.getenv("QWEN_NUM_CTX", "4096"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))  # 25 MB upload limit
MAX_IMAGE_PIXELS = 50_000_000
CONFIDENCE = float(os.getenv("FOODSEG_CONFIDENCE", "0.25"))
FOODSEG_TILE_SIZE = int(os.getenv("FOODSEG_TILE_SIZE", "512"))
FOODSEG_TILE_OVERLAP = int(os.getenv("FOODSEG_TILE_OVERLAP", "64"))

_ollama_endpoint = urlparse(OLLAMA_URL)
if _ollama_endpoint.scheme not in {"http", "https"} or _ollama_endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
    raise RuntimeError("Offline mode requires OLLAMA_URL to point to this computer (localhost)")
if QWEN_MODEL.lower().endswith("-cloud"):
    raise RuntimeError("Offline mode cannot use an Ollama cloud model; configure a locally installed model")

logger = logging.getLogger("foodvision")

# 1. Load PyTorch Segmenter
segmenter, CLASS_NAMES, DEVICE = load_segmenter(SEGMENTER_PATH, LABELS_PATH)
DENSITY_TABLE = json.loads((ROOT / "food_density.json").read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan: checks Ollama connectivity asynchronously without stalling startup."""
    logger.info("Initializing Food Vision service (DeepLabV3+ & Planar Metrology)...")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.5)) as client:
            resp = await client.get(f"{OLLAMA_URL.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                logger.info("Ollama service verified at %s (HTTP 200)", OLLAMA_URL)
            else:
                logger.warning("Ollama connectivity check returned status %s", resp.status_code)
    except Exception as exc:
        logger.info(
            "Ollama service not reachable at %s (%s). Service operates with DeepLabV3+ fallback.",
            OLLAMA_URL,
            exc,
        )
    yield
    logger.info("Shutting down Food Vision service.")


app = FastAPI(title="Food Vision API", version="1.0.0", lifespan=lifespan)

# W3C Compliant CORS Middleware: allow_credentials=False with wildcard origin to prevent preflight rejections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cors_pna_headers(request: Request, call_next):
    """Handle Chrome/Edge Private Network Access preflights and ensure permissive CORS."""
    if request.method == "OPTIONS":
        response = Response(status_code=204)
        origin = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
        response.headers["Access-Control-Allow-Headers"] = request.headers.get("access-control-request-headers", "*")
        if "access-control-request-private-network" in request.headers:
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    response = await call_next(request)
    if "access-control-request-private-network" in request.headers:
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


# Backward-compatibility wrappers
def _segmentation_summary_compat(image: Image.Image):
    return segmentation_summary(image, segmenter, DEVICE, CLASS_NAMES, FOODSEG_TILE_SIZE, FOODSEG_TILE_OVERLAP, CONFIDENCE)


def _build_weight_estimates_compat(components: list[dict[str, Any]], plate_scale: dict[str, Any]):
    return build_weight_estimates(components, plate_scale, DENSITY_TABLE)


def reconcile_clusters_with_vlm(
    clusters: list[dict[str, Any]],
    components: list[dict[str, Any]],
) -> None:
    """
    Reconcile spatial FoodSeg103 clusters with culinary components identified by VLM.
    Assigns cluster_id to components and culinary_name to clusters for SVG visual overlay.
    """
    if not clusters or not components:
        return

    claimed_clusters = set()

    # Pass 1: Explicit cluster_id assignments from VLM
    for comp in components:
        cid = comp.get("cluster_id")
        if cid is not None:
            for cl in clusters:
                if cl["cluster_id"] == cid and cid not in claimed_clusters:
                    claimed_clusters.add(cid)
                    cl["culinary_name"] = comp["name"]
                    break

    # Pass 2: Semantic and spatial area similarity matching
    for comp in components:
        if comp.get("cluster_id") in claimed_clusters:
            continue
        c_name = comp["name"].lower()
        c_frac = float(comp.get("food_area_fraction", 0.0))

        best_score = -1.0
        best_cl = None

        for cl in clusters:
            cid = cl["cluster_id"]
            if cid in claimed_clusters:
                continue

            cl_label = (cl.get("label") or cl.get("class") or "Food").lower()
            cl_frac = float(cl.get("area_fraction", 0.0))

            sim = 0.0
            if c_name == cl_label:
                sim = 1.0
            elif cl_label in c_name or c_name in cl_label:
                sim = 0.75
            else:
                cat_comp = density_category(c_name)
                cat_cl = density_category(cl_label)
                if cat_comp == cat_cl and cat_comp != "mixed_food":
                    sim = 0.50

            area_sim = max(0.0, 1.0 - abs(c_frac - cl_frac))
            total_score = (sim * 0.6) + (area_sim * 0.4)
            if total_score > best_score:
                best_score = total_score
                best_cl = cl

        if best_cl is not None and best_score > 0.20:
            cid = best_cl["cluster_id"]
            claimed_clusters.add(cid)
            comp["cluster_id"] = cid
            best_cl["culinary_name"] = comp["name"]

    for cl in clusters:
        if "culinary_name" not in cl:
            cl["culinary_name"] = (cl.get("label") or cl.get("class") or "Food").title()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "segmenter": "loaded",
        "segmenter_name": "FoodSeg103",
        "segmenter_architecture": "DeepLabV3+ with MobileNetV2 encoder",
        "segmenter_class_count": len(CLASS_NAMES),
        "segmenter_classes": list(CLASS_NAMES.values()),
        "segmenter_device": str(DEVICE),
        "segmenter_input_mode": os.getenv("FOODSEG_STRATEGY", "letterbox-padding-512x512"),
        "segmenter_tile_size": FOODSEG_TILE_SIZE,
        "segmenter_tile_overlap": FOODSEG_TILE_OVERLAP,
        "qwen_context_tokens": QWEN_NUM_CTX,
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "density_categories": list(DENSITY_TABLE),
        "qwen_model": QWEN_MODEL,
        "offline_runtime": True,
    }


async def _read_food_image(upload: UploadFile, field_name: str) -> tuple[Image.Image, bytes]:
    """Validate and decode incoming food image with decompression bomb defense."""
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"{field_name} must be an image")
    raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit",
        )
    try:
        decoded = Image.open(io.BytesIO(raw))
        if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=f"{field_name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit",
            )
        image = ImageOps.exif_transpose(decoded).convert("RGB")
        if image.width < 50 or image.height < 50:
            raise HTTPException(
                status_code=422,
                detail=f"{field_name} resolution is too low ({image.width}x{image.height}). Please provide an image of at least 50x50 pixels.",
            )
        # Check image variance to reject completely blank, uniform, or pitch-black images
        stat = ImageStat.Stat(image)
        if max(stat.var) < 2.0:
            raise HTTPException(
                status_code=422,
                detail=f"{field_name} appears to be completely blank or a single uniform color. Please provide a real photo of food.",
            )
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit",
        )
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail=f"Could not decode {field_name}")
    return image, raw


@app.post("/vision")
async def vision(
    top_image: UploadFile = File(..., description="Photo taken directly above the meal"),
    side_image: UploadFile | None = File(None, description="Photo taken from the side at plate level"),
    plate_diameter_cm: float = Form(..., description="User-selected plate diameter in cm (typical: 20, 25, or 30)"),
    food_hint: str | None = Form(None, description="Optional user culinary note or dish name hint"),
) -> dict[str, Any]:
    """Estimate ingredient weights from paired views, known plate diameter, and spatial prior consensus."""
    if not 15 <= plate_diameter_cm <= 40:
        raise HTTPException(status_code=422, detail="plate_diameter_cm must be between 15 and 40 cm")
    top, top_bytes = await _read_food_image(top_image, "top_image")
    side, side_bytes = None, None
    has_distinct_side = False
    if side_image is not None and getattr(side_image, "filename", None):
        try:
            side, side_bytes = await _read_food_image(side_image, "side_image")
            has_distinct_side = bool(side_bytes and side_bytes != top_bytes)
        except Exception as exc:
            logger.warning("Side image decode failed (%s). Proceeding with top view only.", exc)
            side = None
            has_distinct_side = False

    started = time.perf_counter()

    # 1. Perspective Plate Metrology & Tilt Unwarping
    plate_scale = await anyio.to_thread.run_sync(detect_top_plate_scale, top, plate_diameter_cm, 0)
    top_for_analysis = top
    rectified_homography = False
    M = None

    if plate_scale.get("aspect_ratio", 1.0) < 0.90 and plate_scale.get("ellipse_info"):
        rectified_top, M, _ = await anyio.to_thread.run_sync(
            rectify_perspective_plate, top, plate_scale["ellipse_info"]
        )
        if M is not None:
            top_for_analysis = rectified_top
            rectified_homography = True
            logger.info(
                "Plate perspective tilt rectified using homography (aspect ratio %.2f -> 1.0)",
                plate_scale.get("aspect_ratio", 1.0),
            )

    # 2. DeepLabV3+ Semantic Segmentation with Spatial Cluster & Polygon Extraction
    def _run_top_segmentation():
        return segmentation_summary_with_clusters(
            top_for_analysis, segmenter, DEVICE, CLASS_NAMES, FOODSEG_TILE_SIZE, CONFIDENCE
        )

    def _run_side_segmentation():
        return segmentation_summary(
            side, segmenter, DEVICE, CLASS_NAMES, FOODSEG_TILE_SIZE, FOODSEG_TILE_OVERLAP, CONFIDENCE
        )

    top_areas, top_detections, top_food_pixels, clusters = await anyio.to_thread.run_sync(_run_top_segmentation)
    if side and has_distinct_side:
        side_areas, side_detections, _ = await anyio.to_thread.run_sync(_run_side_segmentation)
    else:
        side_areas, side_detections = [], []

    # Update plate scale with actual food pixel count
    plate_physical_area_cm2 = np.pi * ((plate_diameter_cm / 2.0) ** 2)
    if plate_scale.get("detected"):
        diameter_px = plate_scale.get("plate_diameter_px", 1.0)
        # When homography tilt unwarping is active, the plate on top_for_analysis is unwarped into a circle.
        # Dividing by unrectified minor_px would inflate food coverage by 1/aspect_ratio.
        if rectified_homography:
            plate_area_px = (np.pi / 4.0) * (diameter_px ** 2)
        else:
            minor_px = plate_scale.get("minor_axis_px", diameter_px)
            plate_area_px = (np.pi / 4.0) * diameter_px * minor_px

        food_to_plate_ratio = top_food_pixels / max(1.0, plate_area_px)
        # If food pixels from DeepLabV3+ are near zero but food items were identified, use conservative plate coverage
        effective_coverage = max(0.40, food_to_plate_ratio) if food_to_plate_ratio < 0.05 else min(food_to_plate_ratio, 1.0)
        plate_scale["food_to_plate_ratio"] = round(food_to_plate_ratio, 3)
        plate_scale["food_footprint_cm2"] = round(effective_coverage * plate_physical_area_cm2, 1)
    else:
        # Heuristic fallback for partially visible or cropped plates
        effective_coverage = max(0.45, min(1.0, top_food_pixels / max(1.0, (top.width * top.height) * 0.70)))
        plate_scale["food_to_plate_ratio"] = round(effective_coverage, 3)
        plate_scale["food_footprint_cm2"] = round(effective_coverage * plate_physical_area_cm2, 1)

    # When rectified, project cluster polygon contours back onto original photo coordinates for SVG overlay
    if rectified_homography and M is not None:
        rect_dim = top_for_analysis.width
        for cl in clusters:
            if "polygon_norm" in cl:
                cl["polygon_norm"] = unrectify_polygon_points(
                    cl["polygon_norm"], rect_dim, top.width, top.height, M
                )

    segmented_at = time.perf_counter()

    # 3. Dynamic Multi-Crop Extraction & Segmentation Priors Formatting
    top_thumb_b64 = await anyio.to_thread.run_sync(get_vlm_thumbnail_b64, top_for_analysis, 768)
    side_thumb_b64 = (
        await anyio.to_thread.run_sync(get_vlm_thumbnail_b64, side, 768)
        if (side and has_distinct_side)
        else None
    )
    focal_crops_b64 = await anyio.to_thread.run_sync(extract_focal_crops, top_for_analysis, clusters, 4)
    priors_xml = format_segmentation_priors(clusters)

    # 4. Multi-Crop Qwen VLM Reasoning
    parsed = None
    vlm_failed = False
    try:
        parsed = await query_qwen_vlm(
            top_thumb_b64=top_thumb_b64,
            plate_diameter_cm=plate_diameter_cm,
            ollama_url=OLLAMA_URL,
            qwen_model=QWEN_MODEL,
            side_thumb_b64=side_thumb_b64,
            focal_crops_b64=focal_crops_b64,
            priors_xml=priors_xml,
            food_hint=food_hint,
            num_threads=4,
        )
    except Exception as exc:
        logger.warning(
            "Ollama/Qwen processing skipped or unusable (%s). Falling back to FoodSeg103 geometric segmentation.",
            exc,
        )

    total_pixels = max(1, top.width * top.height)
    food_pixel_fraction = top_food_pixels / total_pixels

    filtered_areas = [
        item for item in top_areas
        if item["class"].lower() not in {"background", "other ingredients"}
        and item["area_percent"] >= 2.0
    ]

    is_non_food = False
    non_food_reason = "No food or meal components detected in image"

    if parsed and parsed.get("is_food") is False:
        is_non_food = True
        non_food_reason = parsed.get("side_view_observation") or "Visual model classified image as non-food"
    elif not parsed:
        if len(filtered_areas) == 0 or food_pixel_fraction < 0.015:
            is_non_food = True
            non_food_reason = f"No food segments detected (food coverage: {food_pixel_fraction * 100:.1f}%)"

    if is_non_food:
        logger.warning(
            "Non-food or non-relevant image detected: %s (coverage=%.4f)",
            non_food_reason, food_pixel_fraction
        )
        return {
            "is_food": False,
            "dish": "Non-Food Image",
            "message": "No food was detected in this image. Please upload a clear photo of food or a plated meal.",
            "reason": non_food_reason,
            "ingredients": {},
            "estimates": [],
            "scale_reference": {"plate_diameter_cm": plate_diameter_cm, **plate_scale},
            "spatial_clusters": [],
            "side_view_observation": "No food detected",
            "total_weight_g": 0,
            "limitations": [
                "The vision system could not detect any recognizable food, edible items, or meal components in this image.",
                "Please ensure the image clearly shows a meal on a plate or in a bowl with adequate lighting."
            ],
            "segmentation": {
                "top_view": {"areas": top_areas, "detections": top_detections},
                "side_view": {"areas": side_areas, "detections": side_detections} if side else None,
                "food_detected": False,
                "food_pixel_fraction": round(food_pixel_fraction, 4),
            },
        }

    if not parsed:
        vlm_failed = True
        clean_components = []
        has_curry_or_meat = any(
            any(m in item["class"].lower() for m in ("chicken", "duck", "pork", "beef", "steak", "meat"))
            for item in filtered_areas
        )

        for item in filtered_areas:
            raw_name = item["class"].lower()
            if raw_name in {"milkshake", "ice cream"}:
                if has_curry_or_meat:
                    comp_name = "Rice"
                else:
                    continue
            elif "chicken" in raw_name or "duck" in raw_name:
                comp_name = "Chicken"
            elif "pork" in raw_name:
                comp_name = "Meat" if has_curry_or_meat else "Pork"
            elif "cilantro" in raw_name or "mint" in raw_name:
                comp_name = "Fresh Herbs"
            elif "onion" in raw_name:
                comp_name = "Onions"
            elif raw_name in {"soup", "sausage", "candy"}:
                continue
            else:
                comp_name = item["class"].title()
            clean_components.append((comp_name, item["area_percent"]))

        merged_dict: dict[str, float] = {}
        for c_name, a_pct in clean_components:
            merged_dict[c_name] = merged_dict.get(c_name, 0.0) + a_pct

        total_area = sum(merged_dict.values()) or 1.0
        components = []
        for c_name, a_pct in merged_dict.items():
            share = a_pct / total_area
            cat = density_category(c_name)
            is_flat = cat in ("bread", "papad")
            components.append({
                "name": c_name,
                "food_area_fraction": round(share, 3),
                "elevation": "flat" if is_flat else "medium",
                "side_view_used": False,
                "height_cm_min": (0.01 if is_flat else 0.05) * plate_diameter_cm,
                "height_cm_max": (0.03 if is_flat else 0.12) * plate_diameter_cm,
                "shape_factor": 0.5 if is_flat else 0.7,
                "volume_ml_min": max(25.0, share * 250.0),
                "volume_ml_max": max(50.0, share * 500.0),
                "confidence": "low",
            })

        sorted_keys = sorted(merged_dict.keys(), key=lambda k: merged_dict[k], reverse=True)
        if not sorted_keys or len(components) == 0:
            return {
                "is_food": False,
                "dish": "Non-Food Image",
                "message": "No food was detected in this image. Please upload a clear photo of food or a plated meal.",
                "reason": "Geometric segmentation produced no valid food components",
                "ingredients": {},
                "estimates": [],
                "scale_reference": {"plate_diameter_cm": plate_diameter_cm, **plate_scale},
                "spatial_clusters": [],
                "side_view_observation": "No food detected",
                "total_weight_g": 0,
                "limitations": [
                    "The vision system could not detect any recognizable food, edible items, or meal components in this image.",
                    "Please ensure the image clearly shows a meal on a plate or in a bowl with adequate lighting."
                ],
                "segmentation": {
                    "top_view": {"areas": top_areas, "detections": top_detections},
                    "side_view": {"areas": side_areas, "detections": side_detections} if side else None,
                    "food_detected": False,
                    "food_pixel_fraction": round(food_pixel_fraction, 4),
                },
            }

        dish_name = f"{sorted_keys[0]} with {sorted_keys[1]}" if len(sorted_keys) >= 2 else f"{sorted_keys[0]} Plate"
        if food_hint and food_hint.strip():
            dish_name = food_hint.strip()

        parsed = {
            "is_food": True,
            "dish": dish_name,
            "components": components,
            "side_view_observation": "VLM reasoning unavailable; geometric segmentation fallback applied.",
        }

    # 5. Spatial-Semantic Consensus Reconciliation
    reconcile_clusters_with_vlm(clusters, parsed["components"])

    # 6. Recipe Disaggregation (Physical Mass Preservation for Submerged Solids)
    parsed["components"] = disaggregate_recipe_components(parsed["components"], parsed.get("dish", ""))

    logger.info(
        "vision completed: segmentation=%.2fs qwen=%.2fs total=%.2fs top=%sx%s side=%s clusters=%d",
        segmented_at - started, time.perf_counter() - segmented_at, time.perf_counter() - started,
        top.width, top.height, f"{side.width}x{side.height}" if side else "none", len(clusters),
    )

    # 7. Dynamic Physical Bounded Weight Estimation
    estimates = await anyio.to_thread.run_sync(build_weight_estimates, parsed["components"], plate_scale, DENSITY_TABLE)

    top_classes = [item["class"] for item in top_areas]
    side_classes = [item["class"] for item in side_areas]
    views_share_class = bool(set(top_classes) & set(side_classes))
    limitations = [
        "These are AI-assisted estimates, not measured weights.",
        "Plate calibration depends on a visible, framed plate rim; perspective compensation is applied when tilted.",
        "The multi-class segmentation model can miss or confuse ingredients, especially when foods overlap.",
        "Food bulk density varies with packing, moisture, cooking method, and recipe.",
    ]
    if rectified_homography:
        limitations.append(
            f"Handheld perspective tilt ({plate_scale.get('tilt_angle_deg', 0):.1f}°) was detected and rectified using homographic unwarping."
        )
    if not has_distinct_side:
        limitations.append(
            "Only the overhead top-view photo was available or distinct. Standard category elevation was assumed without side-view height observation."
        )
    else:
        side_obs_text = parsed.get("side_view_observation", "observed")
        limitations.append(
            f"Dual views analyzed: overhead photo used for 2D footprint; side-view observation ('{side_obs_text}') used to modulate elevation profiles."
        )
    if not plate_scale.get("detected"):
        limitations.append(
            f"Plate rim calibration could not be completed ({plate_scale.get('reason', 'undetected')}); portion volumes rely on uncalibrated heuristic fallbacks."
        )
    if vlm_failed:
        limitations.append(
            "VLM visual reasoning failed or was unavailable; components were estimated from geometric segmentation masks. "
            "Labels represent coarse visual categories, not verified culinary ingredients."
        )
    if top_classes and side_classes and not views_share_class:
        limitations.append(
            "FoodSeg labels differ between the two views (top: " + ", ".join(top_classes[:3]) +
            "; side: " + ", ".join(side_classes[:3]) +
            "). Check that both photos show the same serving; the segmenter can also mislabel food."
        )
        for estimate in estimates:
            if estimate["confidence"] == "high":
                estimate["confidence"] = "medium"

    ingredient_totals: dict[str, int] = {}
    for item in estimates:
        name = item["name"]
        est_w = int(item["weight_g"]["estimate"])
        if est_w <= 0:
            est_w = max(35, int(item["weight_g"].get("min", 25)))
            item["weight_g"]["estimate"] = est_w
            item["weight_g"]["min"] = max(15, int(est_w * 0.7))
            item["weight_g"]["max"] = int(est_w * 1.3)
        ingredient_totals[name] = ingredient_totals.get(name, 0) + est_w

    method_summary = (
        f"Calibrated top-view plate scale ({plate_scale.get('method', 'plate rim')}) combined with "
        f"{'side-view elevation profile' if has_distinct_side else 'standard elevation baseline'} and category bulk density bounding."
        if plate_scale.get("detected")
        else f"Uncalibrated heuristic fallback ({plate_scale.get('reason', 'rim undetected')}) with category bulk density bounding."
    )

    return {
        "is_food": True,
        "dish": parsed["dish"],
        "ingredients": ingredient_totals,
        "estimates": estimates,
        "scale_reference": {"plate_diameter_cm": plate_diameter_cm, **plate_scale},
        "spatial_clusters": clusters,
        "side_view_observation": parsed.get("side_view_observation", ""),
        "image_processing": {
            "top_image_px": {"width": top.width, "height": top.height},
            "side_image_px": {"width": side.width, "height": side.height} if side else None,
            "side_view_provided": has_distinct_side,
            "perspective_rectified": rectified_homography,
            "focal_crops_extracted": len(focal_crops_b64),
            "qwen_input": (
                f"Multi-crop payload (overhead 768px + {len(focal_crops_b64)} focal crops"
                + (" + side elevation)" if has_distinct_side else ")")
            ),
            "segmentation_input": "Native resolution with ImageNet mean padding and continuous logit interpolation",
        },
        "method": method_summary,
        "limitations": limitations,
        "segmentation": {
            "top_view": {"areas": top_areas, "detections": top_detections},
            "side_view": {"areas": side_areas, "detections": side_detections} if side else None,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
