# -*- coding: utf-8 -*-
"""
vision/gemini_router.py
=======================
Multi-Key Round-Robin & Intelligent Failover Router for Google Gemini Vision AI.
Supports seamless key rotation, rate-limit cooldown tracking (429 / Quota),
and structured JSON food recognition for NutriScanAI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

try:
    from vision.vlm import parse_component_response
except ModuleNotFoundError:
    try:
        from vlm import parse_component_response
    except ModuleNotFoundError:
        parse_component_response = None

logger = logging.getLogger("foodvision.gemini")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
KEYS_FILE_DEFAULT = Path(__file__).resolve().parent / "gemini_keys.json"


@dataclass
class GeminiKeyStatus:
    key: str
    status: str = "active"  # "active", "cooldown", "invalid", "error"
    cooldown_until: float = 0.0  # monotonic timestamp
    success_count: int = 0
    failure_count: int = 0
    last_used_at: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def masked(self) -> str:
        clean = self.key.strip()
        if len(clean) <= 10:
            return "******"
        return f"{clean[:6]}...{clean[-4:]}"

    @property
    def is_available(self) -> bool:
        if self.status == "invalid":
            return False
        if self.status == "cooldown":
            return time.monotonic() >= self.cooldown_until
        return True

    @property
    def cooldown_seconds_remaining(self) -> int:
        if self.status != "cooldown":
            return 0
        rem = self.cooldown_until - time.monotonic()
        return max(0, int(rem))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "masked_key": self.masked,
            "status": "active" if self.is_available else self.status,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "cooldown_remaining_sec": self.cooldown_seconds_remaining,
            "last_used_at": self.last_used_at,
            "last_error": self.last_error,
        }


# Strict JSON Schema for Gemini multimodal food breakdown
GEMINI_FOOD_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "is_food": {"type": "BOOLEAN"},
        "dish": {"type": "STRING"},
        "side_view_observation": {"type": "STRING"},
        "components": {
          "type": "ARRAY",
          "items": {
            "type": "OBJECT",
            "properties": {
              "name": {"type": "STRING"},
              "fraction": {"type": "NUMBER"},
              "elevation": {
                "type": "STRING",
                "enum": ["flat", "low", "medium", "heaped"]
              },
              "cluster_id": {"type": "INTEGER"},
              "submerged_ingredients": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
              }
            },
            "required": ["name", "fraction", "elevation"]
          }
        }
    },
    "required": ["is_food", "dish", "components"]
}


class GeminiAllKeysExhaustedError(RuntimeError):
    """Raised when all configured Gemini API keys have failed or are in cooldown."""
    pass


class GeminiKeyRouter:
    """
    Round-Robin & Intelligent Failover Router across multiple Gemini API keys.
    Automatically unfreezes cooling-down keys and rotates requests to maximize
    free-tier throughput and eliminate single-key rate limits.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        keys_file: Optional[Path] = None,
        default_keys: Optional[List[str]] = None,
    ):
        self.model = model
        self.keys_file = Path(keys_file or KEYS_FILE_DEFAULT)
        self._keys: List[GeminiKeyStatus] = []
        self._current_index = 0
        self._lock = asyncio.Lock()

        # Initialize keys from env and file
        self.load_keys(default_keys)

    def load_keys(self, default_keys: Optional[List[str]] = None) -> int:
        """Load keys from environment variables and persistence file."""
        found_keys: List[str] = []

        # 1. Check explicit default keys passed in constructor
        if default_keys:
            found_keys.extend(default_keys)

        # 2. Check GEMINI_API_KEYS (comma, semicolon, or newline separated)
        env_multi = os.getenv("GEMINI_API_KEYS", "")
        if env_multi:
            for item in re.split(r"[,;\n\r\t]+", env_multi):
                item = item.strip()
                if item and item not in found_keys:
                    found_keys.append(item)

        # 3. Check numbered env vars: GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
        for i in range(1, 20):
            k = os.getenv(f"GEMINI_API_KEY_{i}")
            if k and k.strip() and k.strip() not in found_keys:
                found_keys.append(k.strip())

        # 4. Check standard singular GEMINI_API_KEY
        env_single = os.getenv("GEMINI_API_KEY", "")
        if env_single and env_single.strip() and env_single.strip() not in found_keys:
            found_keys.append(env_single.strip())

        # 5. Check persistent keys file
        if self.keys_file.exists():
            try:
                with open(self.keys_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_keys = data.get("keys", []) if isinstance(data, dict) else data
                    if isinstance(file_keys, list):
                        for item in file_keys:
                            if isinstance(item, str) and item.strip() and item.strip() not in found_keys:
                                found_keys.append(item.strip())
            except Exception as e:
                logger.warning("Could not read Gemini keys from %s: %s", self.keys_file, e)

        # Retain existing stats if re-loading
        existing_map = {k.key: k for k in self._keys}
        new_list: List[GeminiKeyStatus] = []
        for raw_k in found_keys:
            clean = raw_k.strip()
            if not clean:
                continue
            if clean in existing_map:
                new_list.append(existing_map[clean])
            else:
                new_list.append(GeminiKeyStatus(key=clean))

        self._keys = new_list
        logger.info("GeminiKeyRouter initialized with %d API key(s).", len(self._keys))
        return len(self._keys)

    def save_keys(self) -> None:
        """Persist current key list to gemini_keys.json."""
        try:
            self.keys_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "keys": [k.key for k in self._keys if k.status != "invalid"],
            }
            with open(self.keys_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("Saved %d Gemini key(s) to %s", len(payload["keys"]), self.keys_file)
        except Exception as e:
            logger.error("Failed to persist Gemini keys to %s: %s", self.keys_file, e)

    def add_key(self, raw_key: str) -> bool:
        """Add a new API key to the router pool."""
        clean = raw_key.strip()
        if not clean or any(k.key == clean for k in self._keys):
            return False
        self._keys.append(GeminiKeyStatus(key=clean))
        self.save_keys()
        logger.info("Added new Gemini key (%s...%s). Total keys: %d", clean[:6], clean[-4:], len(self._keys))
        return True

    def set_keys(self, key_list: List[str]) -> int:
        """Replace or bulk update the API key pool."""
        valid_keys = [k.strip() for k in key_list if isinstance(k, str) and k.strip()]
        unique_keys = list(dict.fromkeys(valid_keys))
        existing_map = {k.key: k for k in self._keys}

        new_keys: List[GeminiKeyStatus] = []
        for raw in unique_keys:
            if raw in existing_map:
                new_keys.append(existing_map[raw])
            else:
                new_keys.append(GeminiKeyStatus(key=raw))

        self._keys = new_keys
        self.save_keys()
        logger.info("Set Gemini keys pool to %d key(s).", len(self._keys))
        return len(self._keys)

    def remove_key(self, identifier: str) -> bool:
        """Remove a key by exact string or masked identifier."""
        clean = identifier.strip()
        orig_len = len(self._keys)
        self._keys = [k for k in self._keys if k.key != clean and k.masked != clean]
        if len(self._keys) < orig_len:
            self.save_keys()
            return True
        return False

    @property
    def has_keys(self) -> bool:
        """Check if any keys are loaded in the router pool."""
        return len(self._keys) > 0

    @property
    def has_available_keys(self) -> bool:
        """Check if at least one key is available (not cooling down or invalid)."""
        return any(k.is_available for k in self._keys)

    def get_pool_status(self) -> Dict[str, Any]:
        """Return public status of all keys in pool without leaking credentials."""
        now = time.monotonic()
        active_count = sum(1 for k in self._keys if k.is_available)
        cooldown_count = sum(1 for k in self._keys if k.status == "cooldown" and now < k.cooldown_until)
        invalid_count = sum(1 for k in self._keys if k.status == "invalid")

        return {
            "model": self.model,
            "total_keys": len(self._keys),
            "active_keys": active_count,
            "cooldown_keys": cooldown_count,
            "invalid_keys": invalid_count,
            "keys": [k.to_dict() for k in self._keys],
        }

    async def get_next_key(self) -> Tuple[GeminiKeyStatus, int]:
        """
        Thread-safe Round-Robin key selector.
        Picks the next available key that is not in cooldown or marked invalid.
        """
        async with self._lock:
            if not self._keys:
                raise GeminiAllKeysExhaustedError(
                    "No Gemini API keys configured. Please add one or more keys via GEMINI_API_KEYS or the Settings UI."
                )

            total = len(self._keys)
            now = time.monotonic()

            # Pass 1: find next available key starting from current index
            for offset in range(total):
                idx = (self._current_index + offset) % total
                cand = self._keys[idx]

                # Check if cooldown has expired
                if cand.status == "cooldown" and now >= cand.cooldown_until:
                    cand.status = "active"
                    cand.cooldown_until = 0.0
                    logger.info("Gemini key %s cooldown expired; restored to active.", cand.masked)

                if cand.is_available:
                    self._current_index = (idx + 1) % total
                    return cand, idx

            # Pass 2: If all keys are in cooldown, locate the one expiring soonest
            cooling = [k for k in self._keys if k.status == "cooldown"]
            if cooling:
                soonest = min(cooling, key=lambda k: k.cooldown_until)
                wait_sec = max(0.5, soonest.cooldown_until - now)
                if wait_sec <= 10.0:
                    logger.info(
                        "All Gemini keys cooling down. Key %s expires in %.1fs. Waiting...",
                        soonest.masked, wait_sec
                    )
                    await asyncio.sleep(wait_sec)
                    soonest.status = "active"
                    soonest.cooldown_until = 0.0
                    return soonest, self._keys.index(soonest)

            raise GeminiAllKeysExhaustedError(
                f"All {total} Gemini API key(s) are currently exhausted or cooling down. "
                "Please add a fresh key or wait for rate limits to reset."
            )

    def report_success(self, key_obj: GeminiKeyStatus) -> None:
        """Record successful invocation for a key."""
        key_obj.status = "active"
        key_obj.success_count += 1
        key_obj.cooldown_until = 0.0
        key_obj.last_used_at = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    def report_failure(
        self,
        key_obj: GeminiKeyStatus,
        status_code: int,
        error_detail: str,
        retry_after: float = 60.0,
    ) -> None:
        """
        Record key failure and adjust status/cooldown based on HTTP status code.
        - 429: Rate limit / Quota exceeded -> 60s cooldown
        - 400 / 403: Invalid API key -> permanently marked invalid
        - 500 / 503 / Network: Transient server failure -> 15s cooldown
        """
        key_obj.failure_count += 1
        key_obj.last_used_at = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        key_obj.last_error = f"HTTP {status_code}: {error_detail[:120]}"

        if status_code == 429 or "RESOURCE_EXHAUSTED" in error_detail or "quota" in error_detail.lower():
            key_obj.status = "cooldown"
            key_obj.cooldown_until = time.monotonic() + max(15.0, retry_after)
            logger.warning(
                "Gemini key %s hit rate limit (HTTP 429). Put on cooldown for %.0fs.",
                key_obj.masked, retry_after
            )
        elif status_code in (400, 403) and ("API_KEY_INVALID" in error_detail or "not valid" in error_detail.lower()):
            key_obj.status = "invalid"
            logger.error("Gemini key %s is invalid or rejected by Google. Marked invalid.", key_obj.masked)
        else:
            key_obj.status = "cooldown"
            key_obj.cooldown_until = time.monotonic() + 15.0
            logger.warning("Gemini key %s encountered error (HTTP %d). Cooldown 15s.", key_obj.masked, status_code)

    async def execute_generate_content(
        self,
        payload: Dict[str, Any],
        timeout: float = 45.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Execute generateContent with automatic Round-Robin key selection and failover.
        Tries keys sequentially until a success occurs or all available keys are exhausted.
        """
        if not self._keys:
            raise GeminiAllKeysExhaustedError("No Gemini API keys configured.")

        max_attempts = len(self._keys)
        tried_keys: set[str] = set()
        last_exception: Optional[Exception] = None

        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=8.0))
            close_client = True

        try:
            for attempt in range(max_attempts):
                try:
                    key_obj, _ = await self.get_next_key()
                except GeminiAllKeysExhaustedError as e:
                    last_exception = e
                    break

                if key_obj.key in tried_keys:
                    continue
                tried_keys.add(key_obj.key)

                url = f"{GEMINI_API_BASE_URL}/models/{self.model}:generateContent?key={key_obj.key}"

                logger.info(
                    "Routing request to Gemini [Model=%s, Key=%s, Attempt=%d/%d]",
                    self.model, key_obj.masked, attempt + 1, max_attempts
                )

                try:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        self.report_success(key_obj)
                        return data, key_obj.masked

                    # Non-200 response -> parse error message
                    err_text = resp.text
                    try:
                        err_json = resp.json()
                        err_text = err_json.get("error", {}).get("message", err_text)
                    except Exception:
                        pass

                    # Extract Retry-After if present
                    retry_after = 60.0
                    ra_hdr = resp.headers.get("retry-after")
                    if ra_hdr and ra_hdr.isdigit():
                        retry_after = float(ra_hdr)

                    self.report_failure(key_obj, resp.status_code, err_text, retry_after)
                    logger.warning(
                        "Gemini key %s failed (HTTP %d: %s). Failing over to next key in pool...",
                        key_obj.masked, resp.status_code, err_text[:100]
                    )

                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as net_err:
                    self.report_failure(key_obj, 503, str(net_err), 15.0)
                    logger.warning(
                        "Gemini network error with key %s (%s). Failing over...",
                        key_obj.masked, net_err
                    )
                    last_exception = net_err

        finally:
            if close_client and client is not None:
                await client.aclose()

        raise GeminiAllKeysExhaustedError(
            f"All {len(tried_keys)} attempted Gemini API key(s) failed. "
            f"Last error: {last_exception}"
        )

    async def analyze_plate(
        self,
        top_image_b64: str,
        plate_diameter_cm: float,
        side_image_b64: Optional[str] = None,
        food_hint: Optional[str] = None,
        priors_xml: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        End-to-End Multimodal Food Identification and Volumetric Decomposition via Gemini.
        Passes overhead view (and optional horizontal side view) with structured JSON Schema.
        """
        has_side = bool(side_image_b64 and side_image_b64 != top_image_b64)

        prompt_parts = [
            f"You are a professional culinary nutrition AI analyzing photos of a plated meal (plate diameter: {plate_diameter_cm:g} cm).",
            "Image 1 is the OVERHEAD (TOP-DOWN) view showing the 2D surface layout and component boundaries.",
        ]
        if has_side:
            prompt_parts.append(
                "Image 2 is the HORIZONTAL (SIDE-LEVEL) view showing vertical food elevation."
            )

        if food_hint and food_hint.strip():
            prompt_parts.append(
                f'\nUser culinary context / dish hint: "{food_hint.strip()}". Use this context to name exact dishes and ingredients.'
            )

        if priors_xml and priors_xml.strip():
            prompt_parts.append(
                f"\nDetected visual segmentation candidates & spatial clusters:\n{priors_xml.strip()}\n"
            )

        prompt_parts.append(
            "\nTasks:\n"
            "1. First verify if the image contains food, a meal, or edible items. "
            "If the photo shows non-food objects (e.g. animals, pets, electronics, vehicles, furniture, empty plate, clothing, documents, random non-food scene), "
            "set 'is_food': false, 'dish': 'Non-Food Item', 'side_view_observation': 'No food detected', 'components': [].\n"
            "2. If food IS present: set 'is_food': true, identify the exact culinary dish name (e.g. Chicken Dum Biryani, Paneer Butter Masala with Naan, Steamed Rice with Dal, Grilled Chicken Salad).\n"
            "3. Identify all distinct food components with their top-view surface area fractions summing to 1.0.\n"
            "4. For each component, specify:\n"
            "   - 'name': culinary name (e.g. Rice, Dal, Chicken Curry, Roti, Salad, Paneer)\n"
            "   - 'fraction': top-view surface area fraction (0.0 to 1.0)\n"
            "   - 'elevation': one of 'flat', 'low', 'medium', 'heaped'\n"
            "   - 'cluster_id': matching cluster ID from visual segmentation priors if applicable\n"
            "   - 'submerged_ingredients': for composite curries/bowls, list submerged solids (e.g. [\"Chicken\"] or [\"Paneer\", \"Capsicum\"])\n"
            "5. If a side view is provided, observe it to assess overall vertical height.\n"
            "Output strictly valid JSON conforming to schema."
        )

        user_parts: List[Dict[str, Any]] = [
            {"text": "\n".join(prompt_parts)},
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": top_image_b64,
                }
            },
        ]
        if has_side and side_image_b64:
            user_parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": side_image_b64,
                }
            })

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": user_parts,
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_FOOD_SCHEMA,
            },
        }

        raw_response, used_key_masked = await self.execute_generate_content(payload, client=client)

        # Extract text content from Gemini candidate
        candidates = raw_response.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned empty candidate response.")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or "text" not in parts[0]:
            raise ValueError("Gemini response did not contain text content.")

        text_content = parts[0]["text"]

        if parse_component_response is not None:
            parsed = parse_component_response(text_content, plate_diameter_cm, has_side_view=has_side)
        else:
            parsed = json.loads(text_content)

        parsed["vlm_provider"] = f"gemini ({self.model})"
        parsed["vlm_key_used"] = used_key_masked
        return parsed

    async def test_connectivity(self) -> Dict[str, Any]:
        """Verify the router and key pool by issuing a lightweight test query."""
        if not self._keys:
            return {
                "status": "error",
                "message": "No Gemini API keys configured in pool. Add keys first.",
                "total_keys": 0,
            }

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Reply with strictly valid JSON: {\"status\": \"ok\", \"service\": \"NutriScanAI Gemini Router\"}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        try:
            raw_resp, used_key = await self.execute_generate_content(payload, timeout=15.0)
            text = raw_resp["candidates"][0]["content"]["parts"][0]["text"]
            return {
                "status": "ok",
                "message": "Gemini Round-Robin router active and responding!",
                "key_used": used_key,
                "model": self.model,
                "response": json.loads(text) if "{" in text else text,
                "pool_status": self.get_pool_status(),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Gemini router test failed: {e}",
                "pool_status": self.get_pool_status(),
            }


# Singleton router instance for the application
gemini_router = GeminiKeyRouter()
