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
                    "cluster_id": {"type": "integer"},
                    "submerged_ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
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


async def query_qwen_vlm(
    top_thumb_b64: str,
    plate_diameter_cm: float,
    ollama_url: str,
    qwen_model: str,
    side_thumb_b64: str | None = None,
    focal_crops_b64: list[str] | None = None,
    priors_xml: str = "",
    food_hint: str | None = None,
    num_threads: int = 4,
) -> Dict[str, Any]:
    """Execute optimized CPU VLM chat inference via Ollama with focal crops, priors, and JSON Schema."""
    has_side = bool(side_thumb_b64 and side_thumb_b64 != top_thumb_b64)
    images = [top_thumb_b64]
    num_focal = len(focal_crops_b64) if focal_crops_b64 else 0
    if focal_crops_b64:
        images.extend(focal_crops_b64)
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

    if priors_xml and priors_xml.strip():
        prompt_parts.append(
            f"\nDetected visual segmentation candidates & spatial clusters:\n{priors_xml.strip()}\n"
        )

    prompt_parts.append(
        "\nTasks:\n"
        "1. First verify if the image contains food, a meal, or edible items. "
        "If the photo shows non-food objects (e.g. animals, pets, electronics, vehicles, furniture, empty plate, clothing, documents, random non-food scene), "
        "output strictly: {\"is_food\": false, \"dish\": \"Non-Food Item\", \"side_view_observation\": \"No food detected in image\", \"components\": []}.\n"
        "2. If food IS present: set 'is_food': true, identify the exact culinary dish name (e.g. Chicken Dum Biryani, Paneer Butter Masala with Naan, Steamed Rice with Dal).\n"
        "3. Identify all distinct food components with their top-view surface area fractions summing to 1.0.\n"
        "4. For each component, assign:\n"
        "   - 'name': culinary name (e.g. Rice, Dal, Chicken Curry, Roti, Salad)\n"
        "   - 'fraction': area fraction (0.0 to 1.0)\n"
        "   - 'elevation': one of 'flat', 'low', 'medium', 'heaped'\n"
        "   - 'cluster_id': matching cluster ID from visual segmentation priors if applicable\n"
        "   - 'submerged_ingredients': for composite dishes or curries, list hidden or submerged solids (e.g. [\"Chicken\"] or [\"Paneer\", \"Capsicum\"])\n"
        "5. If a side view is provided, observe it to assess overall vertical height.\n\n"
        "Output strictly valid JSON matching schema:\n"
        '{"is_food": true, "dish": "Exact Dish Name", "side_view_observation": "Brief elevation observation", '
        '"components": [{"name": "Food Name", "fraction": 0.6, "elevation": "medium", "cluster_id": 1, "submerged_ingredients": []}]}'
    )

    prompt = "\n".join(prompt_parts)

    messages = [
        {"role": "user", "content": prompt, "images": images},
    ]

    payload = {
        "model": qwen_model,
        "messages": messages,
        "stream": False,
        "format": VLM_OUTPUT_SCHEMA,
        "options": {
            "temperature": 0,
            "num_predict": 768,
            "num_ctx": 4096,
            "num_thread": num_threads,
        },
        "keep_alive": "10m",
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            response = await client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RequestError) as net_err:
        logger.warning("Ollama connection failed (%s). Offline fallback will be applied.", net_err)
        raise RuntimeError(f"Ollama connection error: {net_err}") from net_err

    ollama_result = response.json()
    done_reason = ollama_result.get("done_reason", "")
    if done_reason == "length":
        logger.warning("Qwen generation reached token limit (done_reason=length); output may be truncated.")

    message = ollama_result.get("message", {})
    content = message.get("content", "").strip()

    # STRICT RULE: Never interpret internal thinking text as a food-identification response!
    if not content:
        logger.warning(
            "Qwen returned empty content (thinking suppressed or generation aborted). done_reason=%s, thinking_len=%d",
            done_reason, len(message.get("thinking", "")),
        )
        raise ValueError("Qwen returned empty content; model generation failed or was suppressed")

    return parse_component_response(content, plate_diameter_cm, has_side_view=has_side)
