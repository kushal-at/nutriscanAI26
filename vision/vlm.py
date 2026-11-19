# -*- coding: utf-8 -*-
"""
vision/vlm.py
=============
Local Vision-Language Model (Qwen3-VL:2B) client, thread-optimized Ollama payload,
resilient multi-pass JSON extraction, and structured culinary meal component parser.
"""

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from PIL import Image
import httpx

try:
    from vision.estimation import density_category
except ModuleNotFoundError:
    from estimation import density_category

logger = logging.getLogger("foodvision")


def get_vlm_thumbnail_b64(image: Image.Image, max_dim: int = 768) -> str:
    """Create a high-fidelity JPEG thumbnail for accurate local VLM visual processing."""
    thumb = image.copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_focal_crops(
    image: Image.Image,
    clusters: List[Dict[str, Any]],
    max_crops: int = 3,
    crop_size: int = 448,
) -> List[str]:
    """
    Extract native-resolution focal crops centered on high-entropy food clusters directly
    from original image buffer to preserve garnishes, seeds, and micro-ingredients.
    """
    if not clusters:
        return []

    orig_w, orig_h = image.size
    focal_b64_list: List[str] = []

    for cluster in clusters[:max_crops]:
        bbox = cluster.get("bbox_px")
        if not bbox or len(bbox) != 4:
            continue
        y0, x0, y1, x1 = bbox
        bw = max(20, x1 - x0)
        bh = max(20, y1 - y0)

        # 15% adaptive padding margin around cluster
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)

        crop_x0 = max(0, x0 - pad_x)
        crop_y0 = max(0, y0 - pad_y)
        crop_x1 = min(orig_w, x1 + pad_x)
        crop_y1 = min(orig_h, y1 + pad_y)

        if crop_x1 - crop_x0 < 10 or crop_y1 - crop_y0 < 10:
            continue

        cropped = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        cropped = cropped.resize((crop_size, crop_size), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=88)
        focal_b64_list.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    return focal_b64_list


def format_segmentation_priors(clusters: List[Dict[str, Any]]) -> str:
    """Format DeepLabV3+ spatial clusters into formal XML priors for VLM spatial grounding."""
    if not clusters:
        return ""
    lines = ["<segmentation_priors source=\"FoodSeg103_DeepLabV3+\">"]
    for c in clusters[:6]:
        cid = c.get("cluster_id", 1)
        cname = c.get("class", "Food")
        area_pct = c.get("area_percent", 0.0)
        conf = c.get("confidence", 0.5) * 100
        bbox = c.get("bbox_norm", [0, 0, 1, 1])
        lines.append(
            f"  - Cluster #{cid} at [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]: "
            f"Candidate='{cname}' (Area={area_pct:.1f}%, Conf={conf:.1f}%)"
        )
    lines.append("</segmentation_priors>")
    return "\n".join(lines)


ELEVATION_FACTORS: Dict[str, Dict[str, float]] = {
    "flat": {"height_mult": 0.55, "shape_override": 0.45},
    "low": {"height_mult": 0.80, "shape_override": 0.55},
    "medium": {"height_mult": 1.00, "shape_override": 0.65},
    "heaped": {"height_mult": 1.35, "shape_override": 0.85},
    "tall": {"height_mult": 1.35, "shape_override": 0.85},
}

# Strict JSON Schema for Ollama structured output enforcement
VLM_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_food": {"type": "boolean"},
        "dish": {"type": "string"},
        "side_view_observation": {"type": "string"},
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "fraction": {"type": "number"},
                    "elevation": {
                        "type": "string",
                        "enum": ["flat", "low", "medium", "heaped"],
                    },
                },
                "required": ["name", "fraction", "elevation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["is_food", "dish", "components"],
    "additionalProperties": False,
}

VLM_RECOGNITION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_food": {"type": "boolean"},
        "dish": {"type": "string"},
        "visible_foods": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_food", "dish", "visible_foods"],
    "additionalProperties": False,
}

REJECT_CONVERSATIONAL_PREFIXES = (
    "then ", "first ", "second ", "third ", "also ", "finally ",
    "and ", "let's check", "let's ", "check ", "s check", "we see ",
    "there is ", "there are ", "here is ", "based on "
)

# Note: "vegetables" and "flatbreads" are intentionally PERMITTED food categories!
INVALID_NAME_STOPWORDS = {
    "dish", "fraction", "component", "total", "schema", "first", "second", "third",
    "proportion", "observation", "image", "plate", "meal", "food", "ingredients",
    "item", "serving", "layer", "view", "check", "s check", "let's check",
}


def is_valid_food_component_name(name: str) -> bool:
    """Validate that name represents a legitimate food ingredient and not conversational reasoning text."""
    if not isinstance(name, str):
        return False
    clean = name.strip()
    # Strip any enclosing punctuation or quotes
    clean = re.sub(r"^[\"\'\-\*•\s]+|[\"\'\-\*•\s:]+$", "", clean)
    if len(clean) < 2 or len(clean) > 50:
        return False
    clean_lower = clean.lower()

    # Reject conversational sentence starters
    for prefix in REJECT_CONVERSATIONAL_PREFIXES:
        if clean_lower.startswith(prefix) or clean_lower == prefix.strip():
            return False

    # Reject exact generic stopwords (without blocking legitimate foods like vegetables/flatbreads)
    if clean_lower in INVALID_NAME_STOPWORDS:
        return False

    # Must contain at least two letters
    if not re.search(r"[A-Za-z]{2,}", clean):
        return False

    # Reject conversational thinking fragments containing phrases like let's, we have, etc.
    if any(frag in clean_lower for frag in ("let's", "let us", "we have", "i see", "look at", "check:", "observe")):
        return False

    return True


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    """
    Resilient multi-pass JSON extractor handling markdown code fences,
    conversational preambles, trailing text, truncated closing brackets,
    and internal thinking tags without crashing.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    # Pass 0: Strip thinking tags (<think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^<think>.*?\n", "", cleaned, flags=re.DOTALL).strip()

    # Pass 1: Extract from markdown code fences ```json { ... } ``` or ``` { ... } ```
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cand = fenced_match.group(1).strip()
        try:
            val = json.loads(cand)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    # Pass 2: Direct json.loads on cleaned text
    try:
        val = json.loads(cleaned)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    # Pass 3: Locate outermost matching curly braces { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        cand = cleaned[first_brace : last_brace + 1].strip()
        try:
            val = json.loads(cand)
            if isinstance(val, dict):
                return val
        except Exception:
            # Clean trailing commas: e.g. {"name": "Rice",} -> {"name": "Rice"}
            repaired = re.sub(r",\s*([\]}])", r"\1", cand)
            try:
                val = json.loads(repaired)
                if isinstance(val, dict):
                    return val
            except Exception:
                pass

    # Pass 4: Incomplete / truncated JSON repair (reconstruct unbalanced brackets)
    if first_brace != -1:
        truncated = cleaned[first_brace:]
        # Strip trailing unquoted keys or dangling commas
        truncated = re.sub(r',\s*\"[^\"]*\"\s*:\s*[^,}\]]*$', '', truncated)
        truncated = re.sub(r',\s*\"[^\"]*\"?\s*$', '', truncated)
        truncated = re.sub(r',\s*$', '', truncated)

        open_squares = truncated.count('[') - truncated.count(']')
        open_curlies = truncated.count('{') - truncated.count('}')

        repaired = truncated + (']' * max(0, open_squares)) + ('}' * max(0, open_curlies))
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
        try:
            val = json.loads(repaired)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    return None


def parse_component_response(text: str, plate_diameter_cm: float, has_side_view: bool = False) -> Dict[str, Any]:
    """Parse strictly structured JSON into geometric meal components with side-view elevation."""
    value = extract_json_payload(text)

    # Validate parsed JSON structure
    if not isinstance(value, dict):
        logger.error("VLM response could not be parsed as valid JSON: %r", text[:300] if text else "")
        raise ValueError("VLM response did not contain valid structured JSON components")

    # Check for explicit non-food detection from VLM
    if value.get("is_food") is False:
        logger.info("VLM explicitly identified image as NON-FOOD.")
        return {
            "is_food": False,
            "dish": "Non-Food Image",
            "side_view_observation": value.get("side_view_observation", "No food or meal detected in image"),
            "components": [],
        }

    if not isinstance(value.get("components"), list) or not value.get("components"):
        logger.warning("VLM returned empty or missing components array; treating as non-food.")
        return {
            "is_food": False,
            "dish": "Non-Food Image",
            "side_view_observation": "No food components identified",
            "components": [],
        }

    dish = str(value.get("dish", "Plated Meal")).strip()
    if not is_valid_food_component_name(dish):
        dish = "Plated Meal"

    side_obs = str(value.get("side_view_observation", "")).strip()
    if not side_obs:
        side_obs = "Side-view elevation profile observed" if has_side_view else "Side view not provided; standard category baseline assumed"

    components = []
    for item in value["components"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        raw_name = item["name"].strip()
        name = raw_name.title()

        # Strict validation: reject conversational fragments ("Then the...", "Let's check", etc.)
        if not is_valid_food_component_name(name):
            logger.warning("Rejected invalid / conversational component name: %r", raw_name)
            continue

        try:
            area_val = item.get("fraction", item.get("food_area_fraction", 0.3))
            area_share = max(0.01, min(float(area_val), 1.0))

            cat = density_category(name)
            if cat == "bread":
                default_hmin, default_hmax, default_shape = 0.015, 0.035, 0.5
            elif cat == "legume_or_curry":
                default_hmin, default_hmax, default_shape = 0.05, 0.12, 0.8
            elif cat == "cooked_grain":
                default_hmin, default_hmax, default_shape = 0.06, 0.15, 0.85
            elif cat == "cooked_meat":
                default_hmin, default_hmax, default_shape = 0.04, 0.10, 0.75
            elif cat == "papad":
                default_hmin, default_hmax, default_shape = 0.01, 0.02, 0.3
            else:
                default_hmin, default_hmax, default_shape = 0.03, 0.08, 0.65

            raw_elev = str(item.get("elevation", "")).lower().strip()
            if raw_elev in ELEVATION_FACTORS:
                elevation = raw_elev
            elif cat in ("bread", "papad"):
                elevation = "flat"
            else:
                elevation = "medium"

            elev_info = ELEVATION_FACTORS.get(elevation, ELEVATION_FACTORS["medium"])
            elev_mult = elev_info["height_mult"]
            if not has_side_view:
                elev_mult = 1.0

            h_min_ratio = float(item.get("height_fraction_min", default_hmin * elev_mult))
            h_max_ratio = float(item.get("height_fraction_max", default_hmax * elev_mult))
            height_fraction_min = max(0.005, min(h_min_ratio, 0.25))
            height_fraction_max = max(height_fraction_min, min(h_max_ratio, 0.25))
            # Sub-linear plate diameter scaling (exponent 0.35) prevents cubic volume explosion (V ~ D^3)
            # Food mounds spread wider on larger plates rather than growing taller
            d_damp = (plate_diameter_cm / 25.0) ** 0.35
            height_min = max(0.1, min(height_fraction_min * 25.0 * d_damp, 5.0))
            height_max = max(height_min, min(height_fraction_max * 25.0 * d_damp, 6.0))

            base_shape = elev_info["shape_override"] or default_shape
            shape_factor = max(0.2, min(float(item.get("shape_factor", base_shape)), 1.0))

            low = float(item.get("volume_ml_min", max(20.0, area_share * 180.0 * elev_mult)))
            high = float(item.get("volume_ml_max", max(low * 1.3, area_share * 400.0 * elev_mult)))
            if high < 5.0:
                low = max(20.0, area_share * 180.0 * elev_mult)
                high = max(low * 1.3, area_share * 400.0 * elev_mult)
        except (KeyError, TypeError, ValueError):
            continue

        confidence = item.get("confidence", "high")
        if confidence not in ("low", "medium", "high"):
            confidence = "high"

        comp_dict = {
            "name": name,
            "food_area_fraction": area_share,
            "elevation": elevation,
            "side_view_used": has_side_view,
            "height_cm_min": height_min,
            "height_cm_max": height_max,
            "shape_factor": shape_factor,
            "volume_ml_min": low,
            "volume_ml_max": high,
            "confidence": confidence,
        }
        if "cluster_id" in item and isinstance(item["cluster_id"], int):
            comp_dict["cluster_id"] = item["cluster_id"]
        if "estimated_weight_g" in item and isinstance(item["estimated_weight_g"], (int, float)):
            comp_dict["estimated_weight_g"] = float(item["estimated_weight_g"])
        if "estimated_volume_ml" in item and isinstance(item["estimated_volume_ml"], (int, float)):
            comp_dict["estimated_volume_ml"] = float(item["estimated_volume_ml"])
        if "submerged_ingredients" in item and isinstance(item["submerged_ingredients"], list):
            sub_list = [
                str(s).title().strip()
                for s in item["submerged_ingredients"]
                if is_valid_food_component_name(str(s))
            ]
            if sub_list:
                comp_dict["submerged_ingredients"] = sub_list

        components.append(comp_dict)

    if not components:
        logger.info("No valid food components found in VLM response; classifying as non-food.")
        return {
            "is_food": False,
            "dish": "Non-Food Image",
            "side_view_observation": "No food components identified",
            "components": [],
        }

    # Normalize area fractions so they sum to 1.0 within tolerance
    total_area = sum(c["food_area_fraction"] for c in components)
    if total_area > 0:
        for c in components:
            c["food_area_fraction"] = round(c["food_area_fraction"] / total_area, 3)

    return {"is_food": True, "dish": dish, "side_view_observation": side_obs, "components": components}


def parse_recognition_response(text: str) -> Dict[str, Any]:
    """Validate the deliberately simple first-pass culinary recognition response."""
    value = extract_json_payload(text)
    if not isinstance(value, dict):
        raise ValueError("Qwen recognition response did not contain valid JSON")
    if value.get("is_food") is False:
        return {"is_food": False, "dish": "Non-Food Image", "visible_foods": []}

    dish = str(value.get("dish", "Plated Meal")).strip()
    foods = []
    for raw_name in value.get("visible_foods", []):
        name = str(raw_name).strip().title()
        if is_valid_food_component_name(name) and name.lower() not in {item.lower() for item in foods}:
            foods.append(name)
    # Small local models occasionally place the obvious main food only in `dish`.
    # A single-word dish is safe to recover as a visible component (for example, "Idli").
    dish_name = dish.title()
    if len(dish_name.split()) == 1 and is_valid_food_component_name(dish_name) and dish_name.lower() not in {"meal", "food", "plate"}:
        if dish_name.lower() not in {item.lower() for item in foods}:
            foods.insert(0, dish_name)
    if not foods:
        raise ValueError("Qwen recognition response did not identify visible foods")
    return {"is_food": True, "dish": dish if is_valid_food_component_name(dish) else "Plated Meal", "visible_foods": foods}


async def _call_qwen(
    *, prompt: str, images: list[str], output_schema: Dict[str, Any], ollama_url: str,
    qwen_model: str, num_threads: int, max_tokens: int,
) -> str:
    """Run one local structured Qwen request without adding synthetic assistant turns."""
    payload = {
        "model": qwen_model,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
        "think": False,
        "format": output_schema,
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096, "num_thread": num_threads},
        "keep_alive": "10m",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            response = await client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RequestError) as net_err:
        raise RuntimeError(f"Ollama connection error: {net_err}") from net_err

    result = response.json()
    message = result.get("message", {})
    content = message.get("content", "").strip()
    if not content and message.get("thinking"):
        thinking = message["thinking"]
        if "{" in thinking and "}" in thinking:
            content = thinking[thinking.find("{"):thinking.rfind("}") + 1].strip()
    if not content:
        raise ValueError(f"Qwen returned empty content (done_reason={result.get('done_reason', '')})")
    return re.sub(r"<think>[\s\S]*?</think>", "", content).strip()


async def query_qwen_recognition(
    top_thumb_b64: str, ollama_url: str, qwen_model: str, side_thumb_b64: str | None = None,
    food_hint: str | None = None, num_threads: int = 4,
) -> Dict[str, Any]:
    """Identify only visible foods before geometry can bias the culinary answer."""
    images = [top_thumb_b64] + ([side_thumb_b64] if side_thumb_b64 and side_thumb_b64 != top_thumb_b64 else [])
    hint = f' User context: "{food_hint.strip()}".' if food_hint and food_hint.strip() else ""
    prompt = (
        "Identify the foods visibly present in this meal photo. Name only what can be seen; do not infer hidden "
        "ingredients, recipes, or ingredients inside sauces. Treat bowls of chutney, raita, dal, or curry as their "
        "visible dish, not as their unseen contents. For Indian breakfast foods, distinguish a thick white round steamed "
        "rice cake (idli) from a thin browned crepe (dosa): never call a thick white idli a dosa." + hint +
        " Return JSON with is_food, dish, and visible_foods; include the main food in visible_foods."
    )
    return parse_recognition_response(await _call_qwen(
        prompt=prompt, images=images, output_schema=VLM_RECOGNITION_SCHEMA, ollama_url=ollama_url,
        qwen_model=qwen_model, num_threads=num_threads, max_tokens=512,
    ))


async def _call_gemini(
    *, prompt: str, images: list[str], api_key: str, model: str,
) -> str:
    """Call Gemini's multimodal API without persisting credentials or image data."""
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    parts = [{"text": prompt}]
    parts.extend({"inlineData": {"mimeType": "image/jpeg", "data": image}} for image in images)
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
            response = await client.post(
                endpoint,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Gemini API request failed (HTTP {exc.response.status_code})") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Gemini returned no usable recognition response") from exc


async def query_gemini_recognition(
    top_thumb_b64: str, api_key: str, model: str, side_thumb_b64: str | None = None,
    food_hint: str | None = None,
) -> Dict[str, Any]:
    """Use Gemini as the culinary authority for visible food names."""
    images = [top_thumb_b64] + ([side_thumb_b64] if side_thumb_b64 and side_thumb_b64 != top_thumb_b64 else [])
    hint = f' User context: "{food_hint.strip()}".' if food_hint and food_hint.strip() else ""
    prompt = (
        "Identify every distinct, visibly present food item in this Indian meal photo. Name only what can be seen; "
        "do not infer hidden ingredients. Distinguish thick white steamed idli from thin browned dosa. Treat bowls "
        "as their visible dish, such as sambar/dal, raita, or chutney." + hint +
        ' Return JSON only: {"is_food": true, "dish": "short meal name", "visible_foods": ["food name"]}.'
    )
    return parse_recognition_response(await _call_gemini(
        prompt=prompt, images=images, api_key=api_key, model=model,
    ))


async def query_gemini_vlm(
    top_thumb_b64: str, plate_diameter_cm: float, api_key: str, model: str,
    recognized_foods: list[str], side_thumb_b64: str | None = None,
    focal_crops_b64: list[str] | None = None,
) -> Dict[str, Any]:
    """Ask Gemini for visible-food area shares used by the local weight estimator."""
    images = [top_thumb_b64]
    images.extend((focal_crops_b64 or [])[:2])
    if side_thumb_b64 and side_thumb_b64 != top_thumb_b64:
        images.append(side_thumb_b64)
    prompt = (
        "Use the meal photos to estimate surface-area shares for only these already verified visible foods: "
        + ", ".join(recognized_foods)
        + '. Do not add foods or hidden ingredients. Fractions must sum to 1. '
        'Return JSON only: {"is_food": true, "dish": "short meal name", '
        '"side_view_observation": "brief observation", "components": '
        '[{"name": "verified food name", "fraction": 0.25, "elevation": "flat|low|medium|heaped"}]}.'
    )
    return parse_component_response(
        await _call_gemini(prompt=prompt, images=images, api_key=api_key, model=model),
        plate_diameter_cm,
        has_side_view=bool(side_thumb_b64 and side_thumb_b64 != top_thumb_b64),
    )


async def query_qwen_vlm(
    top_thumb_b64: str,
    plate_diameter_cm: float,
    ollama_url: str,
    qwen_model: str,
    side_thumb_b64: str | None = None,
    focal_crops_b64: list[str] | None = None,
    recognized_foods: list[str] | None = None,
    food_hint: str | None = None,
    num_threads: int = 4,
) -> Dict[str, Any]:
    """Measure already-recognized visible foods; segmentation labels never name ingredients."""
    has_side = bool(side_thumb_b64 and side_thumb_b64 != top_thumb_b64)
    images = [top_thumb_b64]
    crops_to_send = (focal_crops_b64[:2] if focal_crops_b64 else [])
    num_focal = len(crops_to_send)
    if crops_to_send:
        images.extend(crops_to_send)
    if has_side:
        images.append(side_thumb_b64)

    prompt_parts = [
        f"You are a professional culinary vision AI analyzing food photos of a plate (plate diameter: {plate_diameter_cm:g} cm).",
        "Image 1 is the OVERHEAD (TOP-DOWN) view showing the complete plate layout and boundaries.",
    ]
    if num_focal > 0:
        prompt_parts.append(
            f"Images 2 through {1 + num_focal} are high-resolution focal crops of specific food clusters detected on the plate."
        )
    if has_side:
        side_idx = 2 + num_focal
        prompt_parts.append(
            f"Image {side_idx} is the HORIZONTAL (SIDE-LEVEL) view showing vertical food elevation."
        )

    if food_hint and food_hint.strip():
        prompt_parts.append(
            f'\nUser culinary context / hint: "{food_hint.strip()}". Use this context to accurately name dishes and ingredients.'
        )

    allowed_foods = ", ".join(recognized_foods or [])

    prompt_parts.append(
        "\nTasks:\n"
        "1. First verify if the image contains food, a meal, or edible items. "
        "If the photo shows non-food objects (e.g. animals, pets, electronics, vehicles, furniture, empty plate, clothing, documents, random non-food scene), "
        "output strictly: {\"is_food\": false, \"dish\": \"Non-Food Item\", \"side_view_observation\": \"No food detected in image\", \"components\": []}.\n"
        "2. If food IS present: set 'is_food': true. Use only visibly present foods from this verified list: " + allowed_foods + ".\n"
        "3. Identify all distinct visible components with their top-view surface area fractions summing to 1.0. Do not add hidden ingredients.\n"
        "4. For each component, assign:\n"
        "   - 'name': culinary name (e.g. Rice, Dal, Chicken Curry, Roti, Salad)\n"
        "   - 'fraction': area fraction (0.0 to 1.0)\n"
        "   - 'elevation': one of 'flat', 'low', 'medium', 'heaped'\n"
        "5. If a side view is provided, observe it to assess overall vertical height.\n\n"
        "Output strictly valid JSON matching schema:\n"
        '{"is_food": true, "dish": "Plated Meal", "side_view_observation": "Brief elevation observation", '
        '"components": [{"name": "Visible Food Name", "fraction": 0.6, "elevation": "medium"}]}'
    )

    prompt = "\n".join(prompt_parts)

    content = await _call_qwen(
        prompt=prompt, images=images, output_schema=VLM_OUTPUT_SCHEMA, ollama_url=ollama_url,
        qwen_model=qwen_model, num_threads=num_threads, max_tokens=1024,
    )
    return parse_component_response(content, plate_diameter_cm, has_side_view=has_side)
