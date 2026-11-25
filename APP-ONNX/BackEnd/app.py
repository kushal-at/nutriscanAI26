# -*- coding: utf-8 -*-
"""
APP/BackEnd/app.py
==================
Main Gateway & Backend Service (Port 8000) for NutriScanAI.
Serves the FrontEnd web application directly and connects to:
- Deep Learning (DL / Vision API) on Port 8001
- Machine Learning (ML / Nutrition API) on Port 8002
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
warnings.filterwarnings("ignore")
logger = logging.getLogger("nutriscan.backend")

BACKEND_DIR = Path(__file__).resolve().parent
APP_DIR = BACKEND_DIR.parent
FRONTEND_DIR = APP_DIR / "FrontEnd"

for p in [str(BACKEND_DIR), str(APP_DIR), str(APP_DIR / "Intelligence")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Service URLs for multi-port architecture
DL_BASE_URL = os.getenv("DL_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")
ML_BASE_URL = os.getenv("ML_SERVICE_URL", "http://127.0.0.1:8002").rstrip("/")
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB

# In-process fallbacks for resilient single-host / deployment fallback
_dl_in_proc_transport = None
_ml_in_proc_transport = None


def _get_dl_in_proc():
    global _dl_in_proc_transport
    if _dl_in_proc_transport is None:
        try:
            from Intelligence.DL.app import app as dl_app
            _dl_in_proc_transport = httpx.ASGITransport(app=dl_app)
        except Exception as exc:
            try:
                from vision.app import app as dl_app
                _dl_in_proc_transport = httpx.ASGITransport(app=dl_app)
            except Exception as exc2:
                logger.error(f"In-process DL app fallback init failed: {exc} / {exc2}")
    return _dl_in_proc_transport


def _get_ml_in_proc():
    global _ml_in_proc_transport
    if _ml_in_proc_transport is None:
        try:
            from Intelligence.ML.app import app as ml_app
            _ml_in_proc_transport = httpx.ASGITransport(app=ml_app)
        except Exception as exc:
            try:
                from ML.app import app as ml_app
                _ml_in_proc_transport = httpx.ASGITransport(app=ml_app)
            except Exception as exc2:
                logger.error(f"In-process ML app fallback init failed: {exc} / {exc2}")
    return _ml_in_proc_transport


async def _check_service(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.0, connect=0.5)) as client:
            r = await client.get(f"{url}/health")
            return r.status_code == 200
    except Exception:
        return False


async def _dispatch_to_dl(method: str, path: str, **kwargs) -> httpx.Response:
    clean_path = path if path.startswith("/") else f"/{path}"
    target_url = f"{DL_BASE_URL}{clean_path}"
    timeout = kwargs.pop("timeout", 180.0)

    # 1. Try external DL service on port 8001
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, target_url, **kwargs)
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        pass

    # 2. In-process fallback
    transport = _get_dl_in_proc()
    if transport is not None:
        async with httpx.AsyncClient(transport=transport, base_url="http://inproc-dl", timeout=timeout) as inproc_c:
            return await inproc_c.request(method, clean_path, **kwargs)

    raise HTTPException(status_code=503, detail=f"Deep Learning (DL) Vision service is unavailable at {target_url}")


async def _dispatch_to_ml(method: str, path: str, **kwargs) -> httpx.Response:
    clean_path = path if path.startswith("/") else f"/{path}"
    target_url = f"{ML_BASE_URL}{clean_path}"
    timeout = kwargs.pop("timeout", 60.0)

    # 1. Try external ML service on port 8002
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, target_url, **kwargs)
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        pass

    # 2. In-process fallback
    transport = _get_ml_in_proc()
    if transport is not None:
        async with httpx.AsyncClient(transport=transport, base_url="http://inproc-ml", timeout=timeout) as inproc_c:
            return await inproc_c.request(method, clean_path, **kwargs)

    raise HTTPException(status_code=503, detail=f"Machine Learning (ML) Nutrition service is unavailable at {target_url}")


async def read_upload_safely(upload_file: UploadFile) -> bytes:
    if upload_file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    raw = await upload_file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{upload_file.filename}' is empty")
    if len(raw) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
        )
    return raw


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="NutriScanAI Gateway & Backend API",
    description="Backend gateway connecting the FrontEnd (Port 8000) to Deep Learning (Port 8001) and Machine Learning (Port 8002).",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# 1. HEALTH & SYSTEM STATUS
# ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="System health check")
async def health_check():
    dl_online = await _check_service(DL_BASE_URL)
    ml_online = await _check_service(ML_BASE_URL)

    return {
        "status": "ok",
        "service": "NutriScanAI Gateway & Backend API",
        "ports": {
            "frontend_backend": 8000,
            "dl_vision": 8001,
            "ml_nutrition": 8002,
        },
        "connected_services": {
            "deep_learning_dl_8001": "online" if dl_online else "standby/in-proc",
            "machine_learning_ml_8002": "online" if ml_online else "standby/in-proc",
        },
    }


@app.get("/api/info", tags=["Health"], summary="API info")
async def api_info():
    return {
        "service": "NutriScanAI Gateway & Backend",
        "version": "5.0.0",
        "architecture": {
            "frontend_backend": "Port 8000 (app.py)",
            "deep_learning_vision": "Port 8001 (Intelligence/DL/app.py)",
            "machine_learning_nutrition": "Port 8002 (Intelligence/ML/app.py)",
        },
    }


# ─────────────────────────────────────────────────────────────
# 2. DUAL-VIEW ANALYSIS ORCHESTRATOR (CONNECTS TO DL & ML)
# ─────────────────────────────────────────────────────────────
@app.post(
    "/analyze/dual",
    tags=["Integration"],
    summary="End-to-End Dual Image Food Analysis",
    description="Forwards dual meal photos to DL Service (8001), extracts portion weights, and predicts nutrition via ML Service (8002).",
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
    }
    if food_hint and food_hint.strip():
        data["food_hint"] = food_hint.strip()

    # Step 1: Query Deep Learning (DL) Vision service on port 8001
    try:
        v_resp = await _dispatch_to_dl("POST", "/vision", files=files, data=data, timeout=180.0)
        v_resp.raise_for_status()
        vision_data = v_resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"DL Vision API error: {exc.response.text[:300]}",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DL Vision communication failure: {exc}")

    dish = vision_data.get("dish")
    ingredients = vision_data.get("ingredients") or {}

    # Handle non-food detection or empty plate
    if vision_data.get("is_food") is False or (not ingredients and not vision_data.get("estimates")):
        return {
            "dish": vision_data.get("dish", "Non-Food Image"),
            "message": vision_data.get("message", "No food was detected in this image. Please upload a clear photo of food."),
            "reason": vision_data.get("reason", "No recognizable food items or plate contents found"),
            "is_food": False,
            "nutrition": [],
            "recipe_totals": {"energy_kcal": 0, "carb_g": 0, "protein_g": 0, "fat_g": 0, "fibre_g": 0},
            "scale_reference": vision_data.get("scale_reference", {}),
            "estimates": [],
            "plate_detected": vision_data.get("plate_detected", False),
            "review_required": True,
            "review_reasons": ["The image could not be verified as a food item."],
            "recognized_visible_foods": [],
            "limitations": vision_data.get("limitations", ["Verification failed: Non-food item or unsegmented view."]),
            "segmentation": vision_data.get("segmentation", {}),
        }

    # Step 2: Query Machine Learning (ML) Nutrition service on port 8002
    ml_payload = {
        "dish": dish or food_hint or None,
        "ingredients": ingredients,
        "estimates": vision_data.get("estimates", []),
    }

    try:
        ml_resp = await _dispatch_to_ml("POST", "/predict/vision", json=ml_payload, timeout=60.0)
        ml_resp.raise_for_status()
        nutrition_data = ml_resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ML Nutrition communication failure: {exc}")

    # Step 3: Merge DL visual results with ML nutrition regression
    return {
        "dish": dish,
        "nutrition": nutrition_data.get("nutrition", []),
        "recipe_totals": nutrition_data.get("recipe_totals", {}),
        "estimates": vision_data.get("estimates", []),
        "scale_reference": vision_data.get("scale_reference", {}),
        "plate_detected": vision_data.get("plate_detected", True),
        "is_food": True,
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
# 3. DIRECT PROXY ENDPOINTS (FORWARDS TO ML:8002 OR DL:8001)
# ─────────────────────────────────────────────────────────────
@app.post("/predict", tags=["Nutrition"], summary="Proxy to ML:8002 /predict")
async def proxy_predict(request: Request):
    body = await request.json()
    resp = await _dispatch_to_ml("POST", "/predict", json=body)
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/predict/vision", tags=["Nutrition"], summary="Proxy to ML:8002 /predict/vision")
async def proxy_predict_vision(request: Request):
    body = await request.json()
    resp = await _dispatch_to_ml("POST", "/predict/vision", json=body)
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.post("/vision", tags=["Vision"], summary="Proxy to DL:8001 /vision")
async def proxy_vision(request: Request):
    form = await request.form()
    files = {}
    data = {}
    for key, val in form.items():
        if hasattr(val, "filename"):
            content = await val.read()
            files[key] = (val.filename, content, val.content_type)
        else:
            data[key] = str(val)

    resp = await _dispatch_to_dl("POST", "/vision", files=files, data=data)
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


@app.api_route("/gemini/{path:path}", methods=["GET", "POST", "DELETE", "PUT"], tags=["Gemini"])
async def proxy_gemini(path: str, request: Request):
    method = request.method
    body = await request.body() if method in ["POST", "PUT"] else None
    params = dict(request.query_params)
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["content-type", "authorization"]}

    resp = await _dispatch_to_dl(method, f"/gemini/{path}", content=body, params=params, headers=headers)
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


# ─────────────────────────────────────────────────────────────
# 4. NUTRIBOT AI ASSISTANT (GEMINI ROUTER + SMART FALLBACK)
# ─────────────────────────────────────────────────────────────
def _run_smart_assistant_fallback(user_text: str, context: dict) -> dict:
    text_lower = user_text.lower().strip()
    profile = context.get("profile", {})
    goals = context.get("goals", {})
    daily = context.get("today", {})

    rem_kcal = daily.get("remainingKcal", goals.get("calories", 2000) - daily.get("calories", 0))
    rem_p = daily.get("remainingProtein", goals.get("protein", 100) - daily.get("protein", 0))
    dietary = goals.get("dietaryPatterns", [])

    if any(k in text_lower for k in ["water", "drank", "drink", "hydrat"]):
        m_l = re.search(r"(\d+(?:\.\d+)?)\s*(?:l|liter|litres)", text_lower)
        m_ml = re.search(r"(\d+)\s*(?:ml|milliliter|cc)", text_lower)
        m_glass = re.search(r"(\d+)\s*(?:glass|glasses|cup|cups)", text_lower)
        m_num = re.search(r"(\d+)", text_lower)

        if m_l:
            amount = int(float(m_l.group(1)) * 1000)
        elif m_ml:
            amount = int(m_ml.group(1))
        elif m_glass:
            amount = int(m_glass.group(1)) * 250
        elif m_num:
            val = int(m_num.group(1))
            amount = val if val > 20 else val * 250
        else:
            amount = 250

        return {
            "message": f"Logged {amount} mL of water to your daily hydration tracker! Staying hydrated supports metabolic rate.",
            "action": "log_water",
            "data": {"amount_ml": amount},
        }

    if any(k in text_lower for k in ["set goal", "update goal", "set target", "protein to", "calories to"]):
        data = {}
        m_prot = re.search(r"protein\s*(?:to|=)?\s*(\d+)", text_lower)
        m_kcal = re.search(r"(?:calorie|kcal|cal)\s*(?:to|=)?\s*(\d+)", text_lower)
        if m_prot:
            data["protein"] = int(m_prot.group(1))
        if m_kcal:
            data["calories"] = int(m_kcal.group(1))

        if data:
            return {
                "message": f"Updated your daily targets: {data}. Changes are saved to your profile.",
                "action": "update_goals",
                "data": data,
            }

    if any(k in text_lower for k in ["suggest", "recommend", "dinner", "lunch", "what should i eat"]):
        dish = "Grilled Lemon-Herb Chicken Bowl with Sweet Potato & Asparagus"
        if "vegetarian" in [d.lower() for d in dietary]:
            dish = "Paneer & Quinoa Bowl with Sautéed Bell Peppers & Avocado Dressing"
        elif "vegan" in [d.lower() for d in dietary]:
            dish = "Crispy Spiced Tofu Bowl with Steamed Edamame, Quinoa & Greens"

        return {
            "message": f"Based on your remaining budget ({max(0, rem_kcal)} kcal & {max(0, rem_p)}g protein), I suggest a **{dish}**.",
            "action": "none",
            "data": {},
        }

    return {
        "message": f"Hello {profile.get('name', 'there')}! You have **{max(0, rem_kcal)} kcal** and **{max(0, rem_p)}g protein** left for today. You can ask for meal ideas or log hydration.",
        "action": "none",
        "data": {},
    }


@app.get("/api/agent/status", tags=["Agent"], summary="Check AI Assistant availability")
async def get_agent_status():
    try:
        resp = await _dispatch_to_dl("GET", "/gemini/keys", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "status": "online",
                "ready": True,
                "provider": "gemini_router",
                "model": data.get("model", "gemini-3-flash-preview"),
                "active_keys": data.get("active_keys", 0),
            }
    except Exception:
        pass
    return {"status": "online", "ready": True, "provider": "gemini_router", "mode": "embedded"}


@app.post("/api/agent/chat", tags=["Agent"], summary="Interact with NutriBot AI Assistant")
async def chat_with_agent(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_messages = body.get("messages", [])
    context = body.get("context", {})

    if not user_messages:
        raise HTTPException(status_code=422, detail="Messages array cannot be empty")

    latest_user_text = ""
    for m in reversed(user_messages):
        if m.get("role") == "user":
            latest_user_text = m.get("content", "").strip()
            break

    profile = context.get("profile", {})
    goals = context.get("goals", {})
    daily = context.get("today", {})

    context_summary = (
        f"User: {profile.get('name', 'User')}. Goal: {profile.get('goal', 'Health')}. "
        f"Daily Targets: {goals.get('calories', 2000)}kcal, {goals.get('protein', 100)}g P. "
        f"Remaining: {daily.get('remainingKcal', 2000)}kcal, {daily.get('remainingProtein', 100)}g P."
    )

    system_instruction = f"""You are NutriBot, an intelligent clinical & sports nutrition assistant for NutriScanAI.
{context_summary}

Respond in concise, valid JSON with:
{{
  "message": "<Warm, conversational response in 1-2 sentences>",
  "action": "none" | "log_water" | "update_goals" | "update_preferences" | "log_meal" | "navigate_view",
  "data": {{ ... }}
}}"""

    chat_messages = []
    for msg in user_messages:
        role = msg.get("role", "user")
        if role in ("user", "assistant"):
            chat_messages.append({"role": role, "content": msg.get("content", "")})

    try:
        resp = await _dispatch_to_dl(
            "POST",
            "/gemini/chat",
            json={
                "system_instruction": system_instruction,
                "messages": chat_messages,
                "temperature": 0.2,
                "max_tokens": 512,
            },
            timeout=25.0,
        )
        if resp.status_code == 200:
            raw_content = (resp.json().get("text") or "{}").strip()
            if raw_content.startswith("```"):
                raw_content = raw_content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                parsed = json.loads(raw_content)
                msg = (parsed.get("message") or "").strip()
                if msg:
                    return {
                        "message": msg,
                        "action": parsed.get("action", "none"),
                        "data": parsed.get("data", {}),
                    }
            except Exception:
                if raw_content:
                    return {"message": raw_content, "action": "none", "data": {}}
    except Exception:
        pass

    return _run_smart_assistant_fallback(latest_user_text, context)


# ─────────────────────────────────────────────────────────────
# 5. STATIC FRONTEND SERVING
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

    @app.get("/style.css", include_in_schema=False)
    async def serve_css():
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

    @app.get("/config.js", include_in_schema=False)
    async def serve_config():
        return FileResponse(FRONTEND_DIR / "config.js", media_type="application/javascript")

    @app.get("/script.js", include_in_schema=False)
    async def serve_js():
        return FileResponse(FRONTEND_DIR / "script.js", media_type="application/javascript")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
