# -*- coding: utf-8 -*-
"""
tests/test_gemini_endpoints.py
==============================
Integration tests for Gemini Round-Robin Key Management Endpoints
in vision/app.py and ML/app.py proxy.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "vision") not in sys.path:
    sys.path.insert(0, str(ROOT / "vision"))
if str(ROOT / "ML") not in sys.path:
    sys.path.insert(0, str(ROOT / "ML"))

from fastapi.testclient import TestClient
from vision.app import app as vision_app
from vision.gemini_router import gemini_router
from ML.app import app as ml_app


def test_vision_health_and_gemini_status():
    client = TestClient(vision_app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "gemini_router" in data
    assert data["gemini_router"]["model"] == gemini_router.model
    print("✓ GET /health reports gemini_router successfully.")


def test_gemini_keys_crud():
    client = TestClient(vision_app)

    # 1. Clear keys for clean testing state
    gemini_router.set_keys([])
    assert len(gemini_router._keys) == 0

    # 2. Add keys via POST /gemini/keys
    resp = client.post("/gemini/keys", json={
        "keys": ["AIzaSy_TEST_KEY_ALPHA_1234", "AIzaSy_TEST_KEY_BRAVO_5678"]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["pool"]["total_keys"] == 2
    assert data["pool"]["active_keys"] == 2
    print("✓ POST /gemini/keys added 2 keys.")

    # 3. GET /gemini/keys
    resp = client.get("/gemini/keys")
    assert resp.status_code == 200
    keys_data = resp.json()
    assert keys_data["total_keys"] == 2
    masked_list = [k["masked_key"] for k in keys_data["keys"]]
    assert any("1234" in m for m in masked_list)
    assert any("5678" in m for m in masked_list)
    # Ensure plaintext is never exposed
    assert not any("ALPHA" in m for m in masked_list)
    print("✓ GET /gemini/keys correctly masked key credentials.")

    # 4. GET & POST /gemini/models
    resp = client.get("/gemini/models")
    assert resp.status_code == 200
    assert len(resp.json()["available_models"]) >= 3
    print("✓ GET /gemini/models returns supported model list.")

    resp = client.post("/gemini/model", json={"model": "gemini-1.5-flash"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "gemini-1.5-flash"
    assert gemini_router.model == "gemini-1.5-flash"
    print("✓ POST /gemini/model switches active model.")

    # Restore default model
    client.post("/gemini/model", json={"model": "gemini-2.5-flash"})

    # 5. DELETE /gemini/keys/{identifier}
    target_masked = masked_list[0]
    resp = client.delete(f"/gemini/keys/{target_masked}")
    assert resp.status_code == 200
    del_data = resp.json()
    assert del_data["pool"]["total_keys"] == 1
    print(f"✓ DELETE /gemini/keys/{target_masked} removed key successfully.")

    # Cleanup
    gemini_router.set_keys([])


def test_ml_proxy_gemini_keys():
    # TestClient for ML app with mock or real vision routing
    client = TestClient(ml_app)
    # Since port 8001 might not be listening during unit test without background server,
    # we verify the route exists in ML app
    routes = [route.path for route in ml_app.routes]
    assert "/gemini/{path:path}" in routes
    print("✓ ML app contains /gemini/{path:path} reverse proxy route.")


if __name__ == "__main__":
    test_vision_health_and_gemini_status()
    test_gemini_keys_crud()
    test_ml_proxy_gemini_keys()
    print("\nALL ENDPOINT TESTS PASSED SUCCESSFULLY!")
