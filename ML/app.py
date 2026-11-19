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
VISION_BASE_URL = os.getenv("VISION_HOST", "http://127.0.0.1:8001").rstrip("/")
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB max upload defense

predictor = NutritionPredictor(MODEL_DIR)

# Backward-compatibility aliases for tests/scripts importing from ML.app
catboost_models = predictor.catboost_models
svd = predictor.svd
vectorizer = predictor.vectorizer

# In-process Vision ASGI transport fallback for unified cloud deployments (Render 512MB / Vercel / Railway)
_in_proc_transport = None

def _get_in_proc_transport():
    global _in_proc_transport
    if _in_proc_transport is None:
        try:
            from vision.app import app as vision_app
            _in_proc_transport = httpx.ASGITransport(app=vision_app)
        except Exception as exc:
            import logging
            logging.getLogger("uvicorn").error(f"In-process vision app initialization failed: {exc}")
    return _in_proc_transport

_external_vision_available = None

async def _check_external_vision() -> bool:
    global _external_vision_available
    if _external_vision_available is not None:
        return _external_vision_available
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.0, connect=0.5)) as test_c:
            r = await test_c.get(f"{VISION_BASE_URL}/health")
            _external_vision_available = (r.status_code == 200)
    except Exception:
        _external_vision_available = False
    return _external_vision_available

async def _dispatch_to_vision(method: str, path: str, **kwargs) -> httpx.Response:
    """
    Dispatch request to external Vision service on port 8001 if available;
    otherwise seamlessly invoke vision_app in-process with ASGITransport.
    """
    global _external_vision_available
    clean_path = path if path.startswith("/") else f"/{path}"
    external_url = f"{VISION_BASE_URL}{clean_path}"
    timeout = kwargs.pop("timeout", 180.0)

    # If external vision port 8001 is known to be offline, bypass network call immediately
    if await _check_external_vision():
        client = getattr(app.state, "http_client", None)
        try:
            if client is not None and not client.is_closed:
                try:
                    return await client.request(method, external_url, timeout=timeout, **kwargs)
                except (httpx.RemoteProtocolError, httpx.WriteError, httpx.ReadError):
                    async with httpx.AsyncClient(timeout=timeout) as retry_client:
                        return await retry_client.request(method, external_url, **kwargs)
            else:
                async with httpx.AsyncClient(timeout=timeout) as temp_client:
                    return await temp_client.request(method, external_url, **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            _external_vision_available = False

    # In-process fallback via ASGITransport
    transport = _get_in_proc_transport()
    if transport is None:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Vision API at {external_url} and in-process fallback unavailable."
        )
    async with httpx.AsyncClient(transport=transport, base_url="http://in-process", timeout=timeout) as in_proc_client:
        return await in_proc_client.request(method, clean_path, **kwargs)


# ─────────────────────────────────────────────────────────────
# 1. LIFESPAN & CONNECTION POOLING
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifespan resources:
    Establishes a shared httpx.AsyncClient with keep-alive, connection pooling, and proper timeouts.
    Syncs Gemini keys from Supabase Cloud Vault on startup.
    """
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0)
    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        app.state.http_client = client

        # Auto-sync Gemini keys from Supabase Cloud Vault into in-process router on startup
        try:
            from vision.gemini_router import gemini_router
            synced = await gemini_router.sync_supabase()
            import logging
            logging.getLogger("uvicorn").info(f"Lifespan: synced {synced} Gemini keys from Supabase Cloud Vault")
        except Exception as exc:
            import logging
            logging.getLogger("uvicorn").warning(f"Lifespan: Supabase key sync error: {exc}")

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

    try:
        v_resp = await _dispatch_to_vision("POST", "/vision", files=files, data=data, timeout=180.0)
        v_resp.raise_for_status()
        vision_data = v_resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Vision API returned error: {exc.response.text[:300]}"
        )
    except HTTPException:
        raise
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
        "review_required": vision_data.get("review_required", False),
        "review_reasons": vision_data.get("review_reasons", []),
        "recognized_visible_foods": vision_data.get("recognized_visible_foods", []),
        "limitations": vision_data.get("limitations", []),
        "segmentation": vision_data.get("segmentation", {}),
        "spatial_clusters": vision_data.get("spatial_clusters", []),
        "vlm_provider": vision_data.get("vlm_provider", "Gemini AI"),
        "vlm_key_used": vision_data.get("vlm_key_used", "N/A"),
        "image_processing": vision_data.get("image_processing", {}),
        "side_view_observation": vision_data.get("side_view_observation", ""),
        "depth_map_b64": vision_data.get("depth_map_b64"),
        "depth_meta": vision_data.get("depth_meta", {}),
    }


# ─────────────────────────────────────────────────────────────
# 3. GEMINI KEY ROUTER & VISION REVERSE PROXY / IN-PROCESS DISPATCH
# ─────────────────────────────────────────────────────────────
@app.api_route(
    "/gemini/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
    tags=["Gemini Router"],
    summary="Proxy Gemini API key management to Vision Service or in-process",
)
async def proxy_gemini_endpoint(path: str, request: Request):
    """Proxy Gemini Key Router management requests to Vision service or in-process."""
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
    try:
        resp = await _dispatch_to_vision(
            request.method,
            f"/gemini/{path}",
            content=body,
            headers=headers,
            params=request.query_params,
            timeout=30.0,
        )
        return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini proxy error: {exc}")


@app.api_route(
    "/vision",
    methods=["POST", "OPTIONS"],
    tags=["Vision"],
    summary="Direct Food Vision API endpoint",
)
async def direct_vision_endpoint(request: Request):
    """Directly proxy or execute /vision requests via port 8001 or in-process."""
    if request.method == "OPTIONS":
        return Response(status_code=204)
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
    try:
        resp = await _dispatch_to_vision(
            "POST",
            "/vision",
            content=body,
            headers=headers,
            params=request.query_params,
            timeout=180.0,
        )
        return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vision error: {exc}")


# ─────────────────────────────────────────────────────────────
# 4. AGENTIC NUTRIBOT CHAT & PREFERENCES ROUTER (OLLAMA QWEN3-VL:2B)
# ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_AGENT_MODEL = os.getenv("OLLAMA_AGENT_MODEL", "qwen3-vl:2b")

@app.get("/api/agent/status", tags=["Agent"], summary="Check Ollama and Qwen3-VL availability")
async def get_agent_status():
    """Verify connectivity to local Ollama server and verify qwen3-vl:2b model availability."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                has_qwen = any("qwen3-vl" in m or "qwen" in m for m in models)
                return {
                    "status": "online",
                    "ollama_url": OLLAMA_BASE_URL,
                    "model": OLLAMA_AGENT_MODEL,
                    "model_available": has_qwen,
                    "installed_models": [m.get("name") for m in data.get("models", [])]
                }
            return {"status": "degraded", "error": f"Ollama HTTP {resp.status_code}"}
    except Exception as exc:
        return {"status": "offline", "error": str(exc), "hint": "Run 'ollama serve' in background"}


@app.post("/api/agent/chat", tags=["Agent"], summary="Interact with Agentic NutriBot via local Qwen3-VL")
async def chat_with_agent(request: Request):
    """
    Agentic assistant powered by local Ollama qwen3-vl:2b.
    Understands user profile, daily macro budget, dietary patterns, and triggers in-app state mutations.
    """
    import json
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_messages = body.get("messages", [])
    context = body.get("context", {})

    if not user_messages:
        raise HTTPException(status_code=422, detail="Messages array cannot be empty")

    # Construct concise contextual agent prompt (optimized for 2B edge model throughput)
    profile = context.get("profile", {})
    goals = context.get("goals", {})
    daily = context.get("today", {})
    recent_meals = context.get("recentMeals", [])

    diet_str = ', '.join(goals.get('dietaryPatterns', [])) or 'None'
    allergens_str = goals.get('allergens', 'None')
    recent_str = ', '.join([m.get('name', 'Meal') for m in recent_meals]) or 'None'

    context_summary = (
        f"User: {profile.get('name', 'User')}, Goal: {profile.get('goal', 'Health')}. "
        f"Daily Targets: {goals.get('calories', 2000)}kcal, {goals.get('protein', 100)}g P, {goals.get('water', 2.5)}L water. "
        f"Consumed: {daily.get('calories', 0)}kcal, {daily.get('protein', 0)}g P, {daily.get('waterMl', 0)}mL water. "
        f"Remaining: {daily.get('remainingKcal', 2000)}kcal, {daily.get('remainingProtein', 100)}g P. "
        f"Diet: {diet_str}. Allergens: {allergens_str}. Recent: {recent_str}."
    )

    system_instruction = f"""You are NutriBot, an AI nutrition assistant for NutriScanAI.
{context_summary}

Respond in concise, valid JSON with:
{{
  "message": "<Warm, conversational response in 1-2 sentences>",
  "action": "none" | "log_water" | "update_goals" | "update_preferences" | "log_meal" | "navigate_view",
  "data": {{ ... }}
}}

Examples:
User: "Drank 300ml water"
JSON: {{"message": "Logged 300 mL of water. Keep staying hydrated!", "action": "log_water", "data": {{"amount_ml": 300}}}}

User: "Set protein to 120g"
JSON: {{"message": "Updated your protein target to 120g.", "action": "update_goals", "data": {{"protein": 120}}}}

User: "What should I eat for dinner?"
JSON: {{"message": "With your remaining calories, a grilled chicken bowl with veggies or steamed tofu and greens would balance your macros.", "action": "none", "data": {{}}}}
"""

    ollama_messages = [{"role": "system", "content": system_instruction}]
    for msg in user_messages:
        role = msg.get("role", "user")
        if role in ("user", "assistant"):
            ollama_messages.append({"role": role, "content": msg.get("content", "")})

    payload = {
        "model": OLLAMA_AGENT_MODEL,
        "messages": ollama_messages,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": 160,
            "temperature": 0.2
        }
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

            raw_content = resp.json().get("message", {}).get("content", "{}")
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = {"message": raw_content.strip(), "action": "none", "data": {}}

            action = parsed.get("action", "none")
            data = parsed.get("data", {})
            msg = (parsed.get("message") or "").strip()

            if not msg:
                if action == "log_water":
                    msg = f"Logged {data.get('amount_ml', 250)} mL of water to your daily tracker!"
                elif action == "update_goals":
                    msg = "Adjusted your nutritional targets."
                elif action == "update_preferences":
                    msg = "Saved your dietary preferences and allergen filters."
                elif action == "log_meal":
                    msg = f"Logged '{data.get('name', 'Meal')}' ({data.get('calories', 0)} kcal) to today's diary."
                elif action == "navigate_view":
                    msg = f"Navigated to {data.get('view', 'dashboard')}."
                else:
                    msg = "I've reviewed your nutritional progress and updated your diary."

            return {
                "message": msg,
                "action": action,
                "data": data,
                "model": OLLAMA_AGENT_MODEL,
            }

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to local Ollama on port 11434. Make sure Ollama is running ('ollama serve')."
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama agent reasoning timed out (took >120s). Please try with a shorter query."
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent reasoning error ({type(exc).__name__}): {exc}")
        raise HTTPException(status_code=500, detail=f"Agent reasoning error: {exc}")


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
