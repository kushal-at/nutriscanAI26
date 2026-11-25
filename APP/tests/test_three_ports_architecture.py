# -*- coding: utf-8 -*-
"""
APP/tests/test_three_ports_architecture.py
==========================================
Integration tests verifying the 3-tier microservice architecture:
- Port 8000: FrontEnd & BackEnd Gateway (BackEnd/app.py)
- Port 8001: Deep Learning Vision API (Intelligence/DL/app.py)
- Port 8002: Machine Learning Nutrition API (Intelligence/ML/app.py)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "BackEnd") not in sys.path:
    sys.path.insert(0, str(ROOT / "BackEnd"))
if str(ROOT / "Intelligence") not in sys.path:
    sys.path.insert(0, str(ROOT / "Intelligence"))

from fastapi.testclient import TestClient
from BackEnd.app import app as backend_app
from Intelligence.DL.app import app as dl_app
from Intelligence.ML.app import app as ml_app


def test_dl_service_port_8001():
    """Verify Deep Learning (DL) service operates on port 8001."""
    client = TestClient(dl_app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "gemini_router" in data


def test_ml_service_port_8002():
    """Verify Machine Learning (ML) service operates on port 8002."""
    client = TestClient(ml_app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["port"] == 8002
    assert "energy_kcal" in data["target_nutrients"]

    # Test prediction
    pred_resp = client.post("/predict", json={
        "ingredients": [
            {"name": "Boiled Rice", "weight_g": 150.0},
            {"name": "Cooked Dal", "weight_g": 100.0}
        ]
    })
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert "ingredients" in pred_data
    assert "recipe_totals" in pred_data
    assert pred_data["recipe_totals"]["energy_kcal"] > 0


def test_backend_gateway_port_8000():
    """Verify BackEnd gateway serves FrontEnd and connects to microservices."""
    client = TestClient(backend_app)

    # 1. Health endpoint
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ports"]["frontend_backend"] == 8000
    assert data["ports"]["dl_vision"] == 8001
    assert data["ports"]["ml_nutrition"] == 8002

    # 2. Serves FrontEnd index
    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "NutriScanAI" in index_resp.text

    # 3. Serves FrontEnd scripts
    js_resp = client.get("/script.js")
    assert js_resp.status_code == 200
    assert "NutriScanAI" in js_resp.text

    # 4. Proxy to ML predict
    ml_proxy_resp = client.post("/predict", json={
        "ingredients": [{"name": "Apple", "weight_g": 100.0}]
    })
    assert ml_proxy_resp.status_code == 200
    assert "recipe_totals" in ml_proxy_resp.json()

    # 5. NutriBot agent status
    agent_resp = client.get("/api/agent/status")
    assert agent_resp.status_code == 200
    assert agent_resp.json()["status"] == "online"
