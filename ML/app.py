# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
🥦 NutraVision Prediction API
==============================
FastAPI service orchestrating nutritional regression predictions, dual-view integration,
and FrontEnd UI static file delivery.
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import httpx
import anyio

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(MODEL_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR.parent))

try:
    from ML.schemas import (
        IngredientIn, NutrientResult, RecipeTotals,
        PredictRequest, PredictResponse,
        VisionRequest, VisionNutrientResult, VisionPredictResponse
    )
    from ML.predictor import NutritionPredictor, PredictionError, TARGETS
except ModuleNotFoundError:
    from schemas import (
        IngredientIn, NutrientResult, RecipeTotals,
        PredictRequest, PredictResponse,
        VisionRequest, VisionNutrientResult, VisionPredictResponse
    )
    from predictor import NutritionPredictor, PredictionError, TARGETS

warnings.filterwarnings("ignore")

FRONTEND_DIR = MODEL_DIR.parent / "FrontEnd"
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://127.0.0.1:8001/vision")
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB max upload defense

predictor = NutritionPredictor(MODEL_DIR)

# Backward-compatibility aliases for tests/scripts importing from ML.app
catboost_models = predictor.catboost_models
svd = predictor.svd
vectorizer = predictor.vectorizer


# ─────────────────────────────────────────────────────────────
# 1. LIFESPAN & CONNECTION POOLING
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifespan resources:
    Establishes a shared httpx.AsyncClient with keep-alive, connection pooling, and proper timeouts.
    """
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0)
    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        app.state.http_client = client
        yield


# ─────────────────────────────────────────────────────────────
# 2. FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="NutraVision Prediction API",
    description=(
        "Predict nutritional values for meals and ingredients using "
        "CatBoost multi-target regression with 3-tier contextual preparation resolution.\n\n"
        "Supported targets: energy_kcal, carb_g, protein_g, fat_g, fibre_g"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.exception_handler(PredictionError)
async def prediction_error_handler(request, exc: PredictionError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def read_upload_safely(upload_file: UploadFile, max_size: int = MAX_UPLOAD_SIZE) -> bytes:
    """Reads upload file in chunks, defending against memory exhaustion by enforcing max_size."""
    if upload_file.size and upload_file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File '{upload_file.filename}' exceeds maximum allowed size of 25 MB."
        )

    buffer = bytearray()
    chunk_size = 1024 * 1024  # 1 MB chunk
    while chunk := await upload_file.read(chunk_size):
        buffer.extend(chunk)
        if len(buffer) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload_file.filename}' exceeds maximum allowed size of 25 MB."
            )
    return bytes(buffer)


# ─────────────────────────────────────────────────────────────
# 3. ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/api/info", tags=["Health"], summary="API info")
async def root():
    return {
        "service": "NutraVision Prediction API",
        "status":  "running",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"], summary="Health check")
async def health():
    return {
        "status":  "ok",
        "models_loaded": list(catboost_models.keys()),
        "svd_components": svd.n_components,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Nutrition"],
    summary="Predict nutritional values for a list of ingredients",
)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        return await anyio.to_thread.run_sync(predictor.predict_ingredients, request.ingredients)
    except (ValueError, PredictionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@app.post(
    "/predict/vision",
    response_model=VisionPredictResponse,
    tags=["Nutrition"],
    summary="Predict nutrition from vision model output",
)
async def predict_vision(request: VisionRequest) -> VisionPredictResponse:
    pairs: List[Tuple[str, float]] = []

    if request.ingredients:
        for name, weight in request.ingredients.items():
            try:
                w_val = float(weight)
                if w_val < 0:
                    raise HTTPException(status_code=422, detail=f"Weight cannot be negative for '{name}': {weight}")
                if w_val == 0:
                    w_val = 50.0
                pairs.append((str(name), w_val))
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid weight for ingredient '{name}': {weight}"
                )
    elif request.estimates:
        for est in request.estimates:
            name = est.get("name", "unknown")
            wg = est.get("weight_g", {})
            if isinstance(wg, dict):
                weight = wg.get("estimate", 100.0)
            else:
                weight = float(wg) if wg else 100.0
            try:
                w_val = float(weight)
                if w_val < 0:
                    raise HTTPException(status_code=422, detail=f"Weight cannot be negative for '{name}': {weight}")
                if w_val == 0:
                    w_val = 50.0
                pairs.append((str(name), w_val))
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid weight for estimate '{name}': {weight}"
                )
    elif request.dish and request.dish.strip():
        pairs.append((request.dish.strip(), 250.0))
    else:
        raise HTTPException(
            status_code=422,
            detail="Request must contain either 'ingredients' dict or 'estimates' array."
        )

    try:
        return await anyio.to_thread.run_sync(predictor.predict_vision_pairs, request.dish, pairs)
    except (ValueError, PredictionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vision prediction error: {exc}")


@app.post(
    "/analyze/dual",
    tags=["Integration"],
    summary="End-to-End Dual Image Food Analysis",
    description="Forwards dual photos to Vision Service (8001), strips extra keys, and predicts nutrition via CatBoost."
)
async def analyze_dual(
    top_image: UploadFile = File(..., description="Top-down overhead photo"),
    side_image: UploadFile = File(None, description="Side-view photo at plate level"),
    plate_diameter_cm: float = Form(25.0, description="Plate diameter in cm (default 25.0)"),
    food_hint: Optional[str] = Form(None, description="Optional user dish or ingredient hint"),
):
    if plate_diameter_cm <= 0:
        raise HTTPException(status_code=422, detail="plate_diameter_cm must be positive")

    top_bytes = await read_upload_safely(top_image)
    side_bytes = await read_upload_safely(side_image) if (side_image and getattr(side_image, "filename", None)) else None

    files = {
        "top_image": (top_image.filename or "top.jpg", top_bytes, top_image.content_type or "image/jpeg"),
    }
    if side_bytes:
        files["side_image"] = (
            side_image.filename or "side.jpg",
            side_bytes,
            side_image.content_type or "image/jpeg",
        )
    data = {
        "plate_diameter_cm": str(plate_diameter_cm),
        "food_hint": food_hint.strip() if food_hint else "",
    }

    # Acquire shared HTTP client with connection pooling and stale keep-alive retry
    client = getattr(app.state, "http_client", None)
    v_resp = None
    try:
        if client is not None and not client.is_closed:
            try:
                v_resp = await client.post(VISION_SERVICE_URL, files=files, data=data)
            except (httpx.RemoteProtocolError, httpx.WriteError, httpx.ReadError):
                # Stale connection in keep-alive pool; retry once on a fresh connection
                async with httpx.AsyncClient(timeout=180.0) as retry_client:
                    v_resp = await retry_client.post(VISION_SERVICE_URL, files=files, data=data)
        else:
            async with httpx.AsyncClient(timeout=180.0) as temp_client:
                v_resp = await temp_client.post(VISION_SERVICE_URL, files=files, data=data)
        v_resp.raise_for_status()
        vision_data = v_resp.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Vision API at {VISION_SERVICE_URL}. Ensure vision service is running on port 8001."
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Vision API returned error: {exc.response.text[:300]}"
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vision communication failure: {exc}")

    dish = vision_data.get("dish")
    ingredients = vision_data.get("ingredients") or {}

    # Edge casing: check if Vision identified image as non-food or if no components exist
    if vision_data.get("is_food") is False or (not ingredients and not vision_data.get("estimates")):
        return {
            "is_food": False,
            "dish": vision_data.get("dish", "Non-Food Image"),
            "message": vision_data.get("message", "No food was detected in this image. Please upload a clear photo of food or a plated meal."),
            "reason": vision_data.get("reason", "No recognizable food items or plate contents found"),
            "ingredients": {},
            "estimates": [],
            "scale_reference": vision_data.get("scale_reference", {}),
            "nutrition": [],
            "recipe_totals": {
                "energy_kcal": 0,
                "carb_g": 0,
                "protein_g": 0,
                "fat_g": 0,
                "fibre_g": 0
            },
            "limitations": vision_data.get("limitations", [
                "No recognizable food or meal was detected in this image."
            ]),
            "segmentation": vision_data.get("segmentation", {})
        }

    pairs: List[Tuple[str, float]] = []
    if ingredients:
        for name, weight in ingredients.items():
            try:
                w_val = float(weight)
                if w_val <= 0:
                    w_val = 50.0
                pairs.append((str(name), w_val))
            except (TypeError, ValueError):
                pass
    elif vision_data.get("estimates"):
        for est in vision_data.get("estimates", []):
            name = est.get("name", "unknown")
            wg = est.get("weight_g", {})
            if isinstance(wg, dict):
                w_val = float(wg.get("estimate", 100.0))
            else:
                w_val = float(wg) if wg else 100.0
            if w_val <= 0:
                w_val = 50.0
            pairs.append((str(name), w_val))
    elif dish and dish.strip():
        pairs.append((dish.strip(), 250.0))

    try:
        nutr = await anyio.to_thread.run_sync(predictor.predict_vision_pairs, dish, pairs)
    except (ValueError, PredictionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return {
        "is_food": True,
        "dish": dish,
        "ingredients": ingredients,
        "estimates": vision_data.get("estimates", []),
        "scale_reference": vision_data.get("scale_reference", {}),
        "nutrition": nutr.nutrition,
        "recipe_totals": nutr.recipe_totals,
        "limitations": vision_data.get("limitations", []),
        "segmentation": vision_data.get("segmentation", {}),
        "spatial_clusters": vision_data.get("spatial_clusters", []),
    }


# ─────────────────────────────────────────────────────────────
# 4. STATIC FRONTEND SERVING
# ─────────────────────────────────────────────────────────────
if FRONTEND_DIR.is_dir():
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    images_dir = FRONTEND_DIR / "images"
    if images_dir.is_dir():
        app.mount("/images", StaticFiles(directory=images_dir), name="images")

    @app.get("/lucide.min.js", include_in_schema=False)
    async def serve_lucide():
        return FileResponse(FRONTEND_DIR / "lucide.min.js", media_type="application/javascript")

    @app.get("/supabase.min.js", include_in_schema=False)
    async def serve_supabase():
        return FileResponse(FRONTEND_DIR / "supabase.min.js", media_type="application/javascript")

    @app.get("/supabase-config.js", include_in_schema=False)
    async def serve_supabase_config():
        return FileResponse(FRONTEND_DIR / "supabase-config.js", media_type="application/javascript")

    @app.get("/sync.js", include_in_schema=False)
    async def serve_sync():
        return FileResponse(FRONTEND_DIR / "sync.js", media_type="application/javascript")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/style.css", include_in_schema=False)
    async def serve_css():
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

    @app.get("/script.js", include_in_schema=False)
    async def serve_js():
        return FileResponse(FRONTEND_DIR / "script.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
