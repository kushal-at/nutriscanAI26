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
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import anyio
import httpx
import numpy as np
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
    from Intelligence.DL.segmentation import (
        load_segmenter,
        segmentation_summary,
        segmentation_summary_with_clusters,
        detect_top_plate_scale,
        rectify_perspective_plate,
        extract_spatial_clusters,
        unrectify_polygon_points,
    )
    from Intelligence.DL.estimation import (
        density_category,
        build_weight_estimates,
        disaggregate_recipe_components,
    )
    from Intelligence.DL.vlm import (
        get_vlm_thumbnail_b64,
        extract_focal_crops,
        format_segmentation_priors,
        query_gemini_recognition,
        query_gemini_vlm,
        query_qwen_recognition,
        parse_component_response,
        query_qwen_vlm,
    )
    from Intelligence.DL.gemini_router import (
        gemini_router,
        GeminiAllKeysExhaustedError,
    )
    from Intelligence.DL.depth import generate_plate_depth_map
except ModuleNotFoundError:
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
            query_gemini_recognition,
            query_gemini_vlm,
            query_qwen_recognition,
            parse_component_response,
            query_qwen_vlm,
        )
        from vision.gemini_router import (
            gemini_router,
            GeminiAllKeysExhaustedError,
        )
        from vision.depth import generate_plate_depth_map
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
            query_gemini_recognition,
            query_gemini_vlm,
            query_qwen_recognition,
            parse_component_response,
            query_qwen_vlm,
        )
        from gemini_router import (
            gemini_router,
            GeminiAllKeysExhaustedError,
        )
        from depth import generate_plate_depth_map

SEGMENTER_PATH = Path(os.getenv("FOODSEG_MODEL", ROOT / "models/FoodSeg103/best_model.onnx"))
LABELS_PATH = Path(os.getenv("FOODSEG_LABELS", ROOT / "models/FoodSeg103/labels_config.json"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl:2b")
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()
QWEN_NUM_CTX = int(os.getenv("QWEN_NUM_CTX", "4096"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))  # 25 MB upload limit
MAX_IMAGE_PIXELS = 50_000_000
CONFIDENCE = float(os.getenv("FOODSEG_CONFIDENCE", "0.25"))
FOODSEG_TILE_SIZE = int(os.getenv("FOODSEG_TILE_SIZE", "512"))
FOODSEG_TILE_OVERLAP = int(os.getenv("FOODSEG_TILE_OVERLAP", "64"))

if VISION_PROVIDER not in {"gemini", "qwen"}:
    VISION_PROVIDER = "gemini"

logger = logging.getLogger("foodvision")

# 1. Load PyTorch Segmenter with Graceful Low-Memory Fallback (for 512MB environments like Render)
segmenter = None
CLASS_NAMES = {}
DEVICE = "cpu"
try:
    skip_heavy = os.getenv("SKIP_HEAVY_SEGMENTER", "false").lower() == "true"
    if not skip_heavy and SEGMENTER_PATH.is_file() and LABELS_PATH.is_file():
        import torch
        torch.set_num_threads(1)
        segmenter, CLASS_NAMES, DEVICE = load_segmenter(SEGMENTER_PATH, LABELS_PATH)
        logger.info("FoodSeg103 DeepLabV3+ loaded successfully on %s", DEVICE)
    else:
        logger.info("Lightweight metrology mode enabled (OpenCV 3D Depth + Gemini AI)")
except Exception as exc:
    logger.warning("Could not load FoodSeg103 segmenter (%s). Falling back to lightweight OpenCV metrology.", exc)
    segmenter = None
    CLASS_NAMES = {}
    DEVICE = "cpu"

DENSITY_TABLE = json.loads((ROOT / "food_density.json").read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan: initializes models, Gemini Key Pool, and Supabase Vault."""
    logger.info("Initializing Food Vision service (DeepLabV3+ & Planar Metrology)...")
    try:
        synced = await gemini_router.sync_supabase()
        if synced > 0:
            logger.info("Synced %d Gemini key(s) from Supabase Cloud Vault.", synced)
    except Exception as exc:
        logger.warning("Supabase key sync deferred: %s", exc)

    status = gemini_router.get_pool_status()
    logger.info(
        "Gemini Round-Robin Key Router online [Model=%s, Total Keys=%d, Active=%d, Vault=%s]",
        gemini_router.model, status["total_keys"], status["active_keys"], status["vault"]["provider"]
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


def _food_name_matches(left: str, right: str) -> bool:
    """Compare culinary labels without treating generic words as evidence of agreement."""
    generic = {"food", "dish", "curry", "sauce", "chutney", "raita", "dal", "gravy", "and", "with"}
    left_terms = {part for part in left.lower().replace("/", " ").split() if part not in generic}
    right_terms = {part for part in right.lower().replace("/", " ").split() if part not in generic}
    return bool(left_terms and right_terms and left_terms.intersection(right_terms)) or left.lower() == right.lower()


def _review_reasons(recognition: dict[str, Any] | None, components: list[dict[str, Any]], vlm_failed: bool) -> list[str]:
    if vlm_failed:
        return ["Culinary recognition was unavailable; the fallback uses coarse visual categories."]
    if not recognition:
        return ["The recognition pass did not return a verified food list."]
    visible = recognition.get("visible_foods", [])
    names = [str(item.get("name", "")) for item in components]
    reasons = []
    if any(not any(_food_name_matches(name, expected) for expected in visible) for name in names):
        reasons.append("The measured components did not fully match the independently recognized visible foods.")
    if any(not any(_food_name_matches(expected, name) for name in names) for expected in visible):
        reasons.append("One or more visible foods could not be assigned a reliable portion estimate.")
    normalized = [name.lower().strip() for name in names]
    if len(normalized) != len(set(normalized)):
        reasons.append("Duplicate food components were detected.")
    return reasons


@app.get("/health")
async def health() -> dict[str, Any]:
    gemini_stat = gemini_router.get_pool_status()
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
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "density_categories": list(DENSITY_TABLE),
        "gemini_router": {
            "model": gemini_stat["model"],
            "total_keys": gemini_stat["total_keys"],
            "active_keys": gemini_stat["active_keys"],
            "cooldown_keys": gemini_stat["cooldown_keys"],
            "has_available_keys": gemini_router.has_available_keys,
        },
        "vision_provider": "gemini",
    }


@app.get("/gemini/keys")
async def get_gemini_keys() -> dict[str, Any]:
    """Return status and masked keys in the Gemini Round-Robin pool."""
    return gemini_router.get_pool_status()


@app.post("/gemini/keys")
async def update_gemini_keys(request: Request) -> dict[str, Any]:
    """
    Add or update keys in the Gemini Round-Robin pool and encrypt into Supabase Vault.
    Body formats accepted:
      {"keys": ["AIzaSy...", ...]}
      {"key": "AIzaSy..."}
      {"action": "replace" | "add", "keys": [...]}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    auth_header = request.headers.get("authorization", "")
    user_token = auth_header.split(" ", 1)[1] if " " in auth_header else None
    user_id = body.get("user_id")

    action = body.get("action", "add")
    keys_input = body.get("keys")
    if keys_input is None and "key" in body:
        keys_input = [body["key"]]
    elif isinstance(keys_input, str):
        keys_input = [k.strip() for k in re.split(r"[,;\n\r\t]+", keys_input) if k.strip()]

    if not isinstance(keys_input, list):
        raise HTTPException(status_code=422, detail="Expected 'keys' list or string")

    added = 0
    if action == "replace":
        gemini_router.set_keys(keys_input)
        for k in keys_input:
            if isinstance(k, str) and k.strip():
                await gemini_router.add_key_async(k.strip(), user_id=user_id, user_token=user_token)
    else:
        for k in keys_input:
            if isinstance(k, str) and k.strip():
                if await gemini_router.add_key_async(k.strip(), user_id=user_id, user_token=user_token):
                    added += 1

    return {
        "status": "ok",
        "message": f"Successfully updated Gemini key pool & encrypted to Supabase (added {added} new key(s)).",
        "pool": gemini_router.get_pool_status(),
    }


@app.delete("/gemini/keys/{identifier}")
async def delete_gemini_key(identifier: str, request: Request) -> dict[str, Any]:
    """Remove a key from the pool and Supabase vault by exact string or masked identifier."""
    auth_header = request.headers.get("authorization", "")
    user_token = auth_header.split(" ", 1)[1] if " " in auth_header else None

    removed = await gemini_router.remove_key_async(identifier, user_token=user_token)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Key '{identifier}' not found in pool.")
    return {
        "status": "ok",
        "message": f"Removed key '{identifier}' from pool and Supabase vault.",
        "pool": gemini_router.get_pool_status(),
    }


@app.post("/gemini/sync")
async def sync_gemini_keys_from_supabase(request: Request) -> dict[str, Any]:
    """Sync and decrypt keys from Supabase Cloud Vault into the active pool."""
    auth_header = request.headers.get("authorization", "")
    user_token = auth_header.split(" ", 1)[1] if " " in auth_header else None
    synced = await gemini_router.sync_supabase(user_token=user_token)
    return {
        "status": "ok",
        "synced_count": synced,
        "message": f"Synced {synced} key(s) from Supabase Cloud Vault.",
        "pool": gemini_router.get_pool_status(),
    }


@app.post("/gemini/test")
async def test_gemini_connectivity() -> dict[str, Any]:
    """Test Gemini Round-Robin router connectivity and failover across configured keys."""
    return await gemini_router.test_connectivity()


@app.get("/gemini/models")
async def get_gemini_models() -> dict[str, Any]:
    """Return available Gemini models for selection."""
    return {
        "current_model": gemini_router.model,
        "available_models": [
            {
                "id": "gemini-3-flash-preview",
                "name": "Gemini 3 Flash Preview",
                "description": "Recommended: latest Google model, fastest multimodal food recognition, native JSON schema."
            },
            {
                "id": "gemini-2.5-flash",
                "name": "Gemini 2.5 Flash",
                "description": "Multimodal culinary recognition with native schema support."
            },
            {
                "id": "gemini-2.5-pro",
                "name": "Gemini 2.5 Pro",
                "description": "Deep reasoning model for complex mixed-ingredient platters and exotic dishes."
            },
        ]
    }


@app.post("/gemini/model")
async def set_gemini_model(request: Request) -> dict[str, Any]:
    """Update active Gemini model."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    model = body.get("model")
    if not model or not isinstance(model, str):
        raise HTTPException(status_code=422, detail="Missing 'model' string")
    gemini_router.model = model.strip()
    return {"status": "ok", "model": gemini_router.model}


@app.post("/gemini/chat")
async def gemini_chat(request: Request) -> dict[str, Any]:
    """Execute text chat via GeminiKeyRouter with round-robin & 429 failover."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])
    system_instruction = (body.get("system_instruction") or "").strip()
    if not messages:
        raise HTTPException(status_code=422, detail="Messages array cannot be empty")

    contents = []
    for m in messages:
        role = "model" if m.get("role") == "assistant" else "user"
        text = str(m.get("content", "")).strip()
        if text:
            contents.append({"role": role, "parts": [{"text": text}]})

    if not contents:
        raise HTTPException(status_code=422, detail="No valid message content provided")

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": float(body.get("temperature", 0.2)),
            "maxOutputTokens": int(body.get("max_tokens", 512)),
            "responseMimeType": "application/json",
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    try:
        raw_resp, key_used = await gemini_router.execute_generate_content(payload, timeout=25.0)
        candidates = raw_resp.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text_out = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        return {
            "status": "ok",
            "text": text_out,
            "key_used": key_used,
            "model": gemini_router.model,
        }
    except GeminiAllKeysExhaustedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini chat error: {exc}")


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
        if segmenter is None:
            return [], [], 0, []
        return segmentation_summary_with_clusters(
            top_for_analysis, segmenter, DEVICE, CLASS_NAMES, FOODSEG_TILE_SIZE, CONFIDENCE
        )

    def _run_side_segmentation():
        if segmenter is None or side is None:
            return [], [], 0
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

    # 3. Generate calibrated 3D Depth & Elevation Map with Turbo colormap
    depth_pil, depth_b64, depth_meta = await anyio.to_thread.run_sync(
        generate_plate_depth_map,
        top_for_analysis,
        plate_scale,
        clusters,
        side if has_distinct_side else None,
        plate_diameter_cm,
        768,
    )

    top_thumb_b64 = await anyio.to_thread.run_sync(get_vlm_thumbnail_b64, top_for_analysis, 768)
    side_thumb_b64 = (
        await anyio.to_thread.run_sync(get_vlm_thumbnail_b64, side, 768)
        if (side and has_distinct_side)
        else None
    )
    focal_crops_b64 = await anyio.to_thread.run_sync(extract_focal_crops, top_for_analysis, clusters, 4)
    # 4. Multimodal culinary analysis via Gemini Round-Robin Router (with dynamic failover)
    parsed = None
    recognition = None
    vlm_failed = False
    priors_xml = format_segmentation_priors(clusters) if clusters else ""

    if gemini_router.has_keys:
        try:
            logger.info(
                "Routing meal analysis to Gemini Key Router (model=%s, %d keys in pool)",
                gemini_router.model, len(gemini_router._keys)
            )
            parsed = await gemini_router.analyze_plate(
                top_image_b64=top_thumb_b64,
                plate_diameter_cm=plate_diameter_cm,
                side_image_b64=side_thumb_b64,
                depth_map_b64=depth_b64,
                food_hint=food_hint,
                priors_xml=priors_xml,
            )
            recognition = {
                "is_food": parsed.get("is_food", True),
                "dish": parsed.get("dish", "Meal"),
                "visible_foods": [c["name"] for c in parsed.get("components", [])],
            }
            logger.info(
                "Gemini analysis successful [Dish=%s, Key=%s, Components=%d]",
                parsed.get("dish"), parsed.get("vlm_key_used"), len(parsed.get("components", []))
            )
        except GeminiAllKeysExhaustedError as exc:
            logger.warning("All Gemini API keys exhausted or rate-limited: %s. Falling back to geometric segmentation.", exc)
            vlm_failed = True
        except Exception as exc:
            logger.error("Gemini router invocation failed: %s. Falling back to geometric segmentation.", exc)
            vlm_failed = True
    elif VISION_PROVIDER == "qwen":
        try:
            recognition = await query_qwen_recognition(
                top_thumb_b64=top_thumb_b64,
                ollama_url=OLLAMA_URL,
                qwen_model=QWEN_MODEL,
                side_thumb_b64=side_thumb_b64,
                food_hint=food_hint,
                num_threads=4,
            )
            if recognition.get("is_food") is False:
                parsed = {**recognition, "components": [], "side_view_observation": "No food detected"}
            else:
                parsed = await query_qwen_vlm(
                    top_thumb_b64=top_thumb_b64,
                    plate_diameter_cm=plate_diameter_cm,
                    ollama_url=OLLAMA_URL,
                    qwen_model=QWEN_MODEL,
                    side_thumb_b64=side_thumb_b64,
                    focal_crops_b64=focal_crops_b64,
                    recognized_foods=recognition["visible_foods"],
                    food_hint=food_hint,
                    num_threads=4,
                )
        except Exception as exc:
            logger.warning("Qwen processing skipped: %s. Falling back to FoodSeg103.", exc)
            vlm_failed = True
    else:
        logger.warning("No Gemini API keys configured. Set keys via Settings UI or GEMINI_API_KEYS. Falling back to FoodSeg103.")
        vlm_failed = True

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

    # 5. Spatial cluster reconciliation is display-only; it cannot rename Qwen components.
    reconcile_clusters_with_vlm(clusters, parsed["components"])

    # Never manufacture hidden recipe ingredients from a photograph. The visible dish name is
    # passed directly to the nutrition model, which avoids false items such as "Rice in Chutney".
    review_reasons = _review_reasons(recognition, parsed["components"], vlm_failed)
    review_required = bool(review_reasons)

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
    if review_required:
        limitations.append("Review the detected foods before logging: " + " ".join(review_reasons))
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
        "scale_reference": {"plate_diameter_cm": plate_diameter_cm, "plate_radius_cm": round(plate_diameter_cm / 2.0, 1), **plate_scale},
        "spatial_clusters": clusters,
        "side_view_observation": parsed.get("side_view_observation", ""),
        "depth_map_b64": depth_b64,
        "depth_meta": depth_meta,
        "vlm_provider": parsed.get("vlm_provider", "DeepLabV3+ Geometric Fallback" if vlm_failed else f"Gemini ({gemini_router.model})"),
        "vlm_key_used": parsed.get("vlm_key_used", "N/A"),
        "image_processing": {
            "top_image_px": {"width": top.width, "height": top.height},
            "side_image_px": {"width": side.width, "height": side.height} if side else None,
            "side_view_provided": has_distinct_side,
            "perspective_rectified": rectified_homography,
            "focal_crops_extracted": len(focal_crops_b64),
            "vlm_provider": parsed.get("vlm_provider", "DeepLabV3+ Geometric Fallback" if vlm_failed else f"Gemini ({gemini_router.model})"),
            "vlm_key_used": parsed.get("vlm_key_used", "N/A"),
            "segmentation_input": "Native resolution with ImageNet mean padding and continuous logit interpolation",
        },
        "method": method_summary,
        "review_required": review_required,
        "review_reasons": review_reasons,
        "recognized_visible_foods": recognition.get("visible_foods", []) if recognition else [],
        "limitations": limitations,
        "segmentation": {
            "top_view": {"areas": top_areas, "detections": top_detections},
            "side_view": {"areas": side_areas, "detections": side_detections} if side else None,
        },
    }


# Hugging Face Spaces (Gradio SDK, no Docker) support
if os.getenv("SPACE_ID") or os.getenv("ENABLE_GRADIO") == "1":
    try:
        import gradio as gr
        import json as _json

        def _hf_status() -> str:
            return _json.dumps({
                "status": "ok",
                "service": "Food Vision API (DL)",
                "segmenter_loaded": segmenter is not None,
                "gemini_keys_active": sum(1 for k in gemini_router._keys if k.is_available),
                "gemini_model": gemini_router.model,
            }, indent=2)

        with gr.Blocks(title="NutriScan DL Vision API") as demo:
            gr.Markdown("# 📸 NutriScan DL — Food Vision & Segmentation API\nFastAPI endpoints are live at `/vision`, `/health`, `/gemini/keys`, and `/docs`.")
            out = gr.Code(label="Service Status", value=_hf_status(), language="json")
            btn = gr.Button("Refresh Status", variant="primary")
            btn.click(_hf_status, outputs=out)

        app = gr.mount_gradio_app(app, demo, path="/")
    except ImportError:
        pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860" if os.getenv("SPACE_ID") else "8001")))
    uvicorn.run(app, host="0.0.0.0", port=port)
