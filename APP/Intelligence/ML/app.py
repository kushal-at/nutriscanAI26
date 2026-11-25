# -*- coding: utf-8 -*-
"""
APP/Intelligence/ML/app.py
==========================
Dedicated Machine Learning Microservice (Port 8002) for NutraVision.
Serves CatBoost nutritional regression predictions, Atwater factor reconciliation,
and mass conservation calculations.
"""

from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent.parent

for p in [str(MODEL_DIR), str(MODEL_DIR.parent), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Intelligence.ML.schemas import (
        IngredientIn,
        NutrientResult,
        RecipeTotals,
        PredictRequest,
        PredictResponse,
        VisionRequest,
        VisionNutrientResult,
        VisionPredictResponse,
    )
    from Intelligence.ML.predictor import NutritionPredictor, PredictionError, TARGETS
except ModuleNotFoundError:
    try:
        from schemas import (
            IngredientIn,
            NutrientResult,
            RecipeTotals,
            PredictRequest,
            PredictResponse,
            VisionRequest,
            VisionNutrientResult,
            VisionPredictResponse,
        )
        from predictor import NutritionPredictor, PredictionError, TARGETS
    except ModuleNotFoundError:
        from ML.schemas import (
            IngredientIn,
            NutrientResult,
            RecipeTotals,
            PredictRequest,
            PredictResponse,
            VisionRequest,
            VisionNutrientResult,
            VisionPredictResponse,
        )
        from ML.predictor import NutritionPredictor, PredictionError, TARGETS

predictor = NutritionPredictor(MODEL_DIR)

# Backward-compatibility aliases
catboost_models = predictor.catboost_models
svd = predictor.svd
vectorizer = predictor.vectorizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="NutraVision ML Nutrition API",
    description="Microservice providing multi-target CatBoost nutritional regression predictions.",
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


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check():
    """Verify ML models are loaded and operational."""
    loaded_models = [k for k, v in predictor.catboost_models.items() if v is not None]
    return {
        "status": "ok",
        "service": "NutraVision ML Nutrition API",
        "port": 8002,
        "models_loaded": len(loaded_models),
        "target_nutrients": TARGETS,
        "preprocessor": {
            "categories": len(predictor.category_keywords),
            "preparations": len(predictor.preparation_keywords),
            "attributes": len(predictor.attribute_names),
        },
    }


@app.get("/api/info", tags=["Health"], summary="API info")
async def api_info():
    return {
        "service": "NutraVision ML Nutrition API",
        "version": "5.0.0",
        "port": 8002,
        "targets": TARGETS,
        "model_type": "CatBoost Multi-Target Regressors (energy, carb, protein, fat, fibre)",
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
                    detail=f"Invalid weight for ingredient '{name}': {weight}",
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
                    detail=f"Invalid weight for estimate '{name}': {weight}",
                )
    elif request.dish and request.dish.strip():
        pairs.append((request.dish.strip(), 250.0))
    else:
        raise HTTPException(
            status_code=422,
            detail="Request must contain either 'ingredients' dict or 'estimates' array.",
        )

    try:
        return await anyio.to_thread.run_sync(predictor.predict_vision_pairs, request.dish, pairs)
    except (ValueError, PredictionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vision prediction error: {exc}")


# Hugging Face Spaces (Gradio SDK, no Docker) support
if os.getenv("SPACE_ID") or os.getenv("ENABLE_GRADIO") == "1":
    try:
        import gradio as gr
        import json as _json

        def _hf_predict(ingredients_json: str) -> str:
            try:
                items = _json.loads(ingredients_json)
                pairs = [(str(i["name"]), float(i.get("weight_g", 100))) for i in items]
                res = predictor.predict_vision_pairs(None, pairs)
                return _json.dumps(res.model_dump(), indent=2)
            except Exception as exc:
                return _json.dumps({"error": str(exc)}, indent=2)

        with gr.Blocks(title="NutriScan ML Nutrition API") as demo:
            gr.Markdown("# 🥗 NutriScan ML — Nutrition Regression API\nFastAPI endpoints are live at `/predict`, `/predict/vision`, `/health`, and `/docs`.")
            inp = gr.Textbox(
                label="Ingredients JSON",
                value='[{"name": "Boiled Rice", "weight_g": 150}, {"name": "Cooked Dal", "weight_g": 100}]',
                lines=3,
            )
            out = gr.Code(label="Predicted Nutrition", language="json")
            btn = gr.Button("Predict Nutrition", variant="primary")
            btn.click(_hf_predict, inputs=inp, outputs=out)

        app = gr.mount_gradio_app(app, demo, path="/")
    except ImportError:
        pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860" if os.getenv("SPACE_ID") else "8002")))
    uvicorn.run(app, host="0.0.0.0", port=port)
