# -*- coding: utf-8 -*-
"""
Tests for GeminiKeyRouter: Round-Robin rotation and automatic failover.
"""

import asyncio
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from Intelligence.DL.gemini_router import GeminiKeyRouter, GeminiKeyStatus, GeminiAllKeysExhaustedError


def test_key_masking():
    k = GeminiKeyStatus(key="AIzaSyA1234567890XYZW_TestKey")
    assert k.masked == "AIzaSy...tKey"
    assert "1234567890" not in k.masked


@pytest.mark.asyncio
async def test_round_robin_rotation():
    router = GeminiKeyRouter()
    router.set_keys(["KEY_AAA_1111", "KEY_BBB_2222", "KEY_CCC_3333"])
    assert len(router._keys) == 3

    k1, idx1 = await router.get_next_key()
    assert k1.key == "KEY_AAA_1111"

    k2, idx2 = await router.get_next_key()
    assert k2.key == "KEY_BBB_2222"

    k3, idx3 = await router.get_next_key()
    assert k3.key == "KEY_CCC_3333"

    # Wraps around to first key
    k4, idx4 = await router.get_next_key()
    assert k4.key == "KEY_AAA_1111"


@pytest.mark.asyncio
async def test_failover_on_429():
    router = GeminiKeyRouter(default_keys=["AIzaSy_KEY_1111", "AIzaSy_KEY_2222", "AIzaSy_KEY_3333"])

    # Mock client where KEY_1 returns 429, but KEY_2 returns 200
    call_counts = {"KEY_1": 0, "KEY_2": 0}

    async def mock_post(url, json, headers):
        if "KEY_1" in url:
            call_counts["KEY_1"] += 1
            resp = MagicMock()
            resp.status_code = 429
            resp.text = '{"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}'
            resp.json.return_value = {"error": {"code": 429, "message": "Resource exhausted"}}
            resp.headers = {"retry-after": "60"}
            return resp
        elif "KEY_2" in url:
            call_counts["KEY_2"] += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "candidates": [{
                    "content": {"parts": [{"text": '{"is_food": true, "dish": "Test Salad", "components": []}'}]}
                }]
            }
            resp.headers = {}
            return resp
        raise RuntimeError("Unexpected key")

    mock_client = AsyncMock()
    mock_client.post = mock_post

    result, used_key = await router.execute_generate_content(
        {"contents": []}, client=mock_client
    )

    # Verified: KEY_1 was called, failed, put on cooldown, and KEY_2 took over and succeeded!
    assert call_counts["KEY_1"] == 1
    assert call_counts["KEY_2"] == 1
    assert "2222" in used_key
    assert result["candidates"][0]["content"]["parts"][0]["text"] is not None

    # Next call should now automatically route to KEY_3 because KEY_1 is on cooldown
    k_next, _ = await router.get_next_key()
    assert "3333" in k_next.key


@pytest.mark.asyncio
async def test_all_keys_exhausted():
    router = GeminiKeyRouter(default_keys=["BAD_KEY"])

    async def mock_post(url, json, headers):
        resp = MagicMock()
        resp.status_code = 403
        resp.text = '{"error": {"message": "API key not valid"}}'
        resp.json.return_value = {"error": {"message": "API key not valid"}}
        resp.headers = {}
        return resp

    mock_client = AsyncMock()
    mock_client.post = mock_post

    with pytest.raises(GeminiAllKeysExhaustedError):
        await router.execute_generate_content({"contents": []}, client=mock_client)


if __name__ == "__main__":
    async def run_all():
        test_key_masking()
        await test_round_robin_rotation()
        await test_failover_on_429()
        await test_all_keys_exhausted()
        print("ALL TESTS PASSED SUCCESSFULLY!")

    asyncio.run(run_all())
