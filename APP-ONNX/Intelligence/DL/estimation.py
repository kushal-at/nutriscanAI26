# -*- coding: utf-8 -*-
"""
vision/estimation.py
====================
Food density category mapping, volumetric modeling (mounded spherical cap,
flat sheet, chunky packing, shallow layers), dynamic physical portion bounding,
and calibrated volume-to-gram weight estimation.
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger("vision.estimation")

RECIPE_DECOMPOSER_PATH = Path(__file__).resolve().parent / "recipe_decomposer.json"
_RECIPE_CACHE = None


def get_recipe_decomposer_data() -> Dict[str, Any]:
    global _RECIPE_CACHE
    if _RECIPE_CACHE is not None:
        return _RECIPE_CACHE
    if RECIPE_DECOMPOSER_PATH.exists():
        try:
            with open(RECIPE_DECOMPOSER_PATH, "r", encoding="utf-8") as f:
                _RECIPE_CACHE = json.load(f)
                return _RECIPE_CACHE
        except Exception as e:
            logger.warning("Failed to load recipe decomposer data: %s", e)
    return {}


def disaggregate_recipe_components(
    components: List[Dict[str, Any]],
    dish_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Decompose composite dishes (e.g. curries, biryanis, bowls) into physical
    sub-constituents (e.g. sauce vs. submerged meat/paneer/legumes) to ensure
    submerged macronutrients and proteins receive accurate volumetric mass.
    """
    decomposer_rules = get_recipe_decomposer_data()
    if not decomposer_rules and not any("submerged_ingredients" in c for c in components):
        return components

    existing_names_lower = {str(c.get("name", "")).lower() for c in components}
    disaggregated: List[Dict[str, Any]] = []

    for item in components:
        name = str(item.get("name", "Food")).strip()
        name_lower = name.lower()
        submerged = item.get("submerged_ingredients")
        area_frac = float(item.get("food_area_fraction", 0.0))

        # Check explicit submerged ingredients from VLM
        if submerged and isinstance(submerged, list) and len(submerged) > 0 and area_frac > 0.02:
            num_subs = len(submerged)
            sub_total_share = 0.40
            gravy_share = 0.60
            per_sub_share = (sub_total_share / num_subs)

            # Gravy/base component
            gravy_comp = dict(item)
            gravy_comp["name"] = f"{name} (Gravy)" if "gravy" not in name_lower and "curry" not in name_lower else name
            gravy_comp["food_area_fraction"] = round(area_frac * gravy_share, 4)
            gravy_comp["submerged_ingredients"] = []
            disaggregated.append(gravy_comp)

            # Submerged solid components
            for sub_name in submerged:
                sub_comp = dict(item)
                sub_comp["name"] = f"{sub_name} (in {name})"
                sub_comp["food_area_fraction"] = round(area_frac * per_sub_share, 4)
                sub_comp["submerged_ingredients"] = []
                sub_comp["submerged"] = True
                disaggregated.append(sub_comp)
            continue

        # Check matched archetype from recipe_decomposer.json
        matched_rule = None
        for rule_id, rule_data in decomposer_rules.items():
            keywords = rule_data.get("match_keywords", [])
            if any(kw in name_lower for kw in keywords):
                sub_comps = rule_data.get("components", [])
                primary_name = sub_comps[0]["name"].lower() if sub_comps else ""
                if not any(primary_name in ex for ex in existing_names_lower):
                    matched_rule = rule_data
                    break

        if matched_rule and area_frac > 0.03:
            rule_comps = matched_rule.get("components", [])
            for rc in rule_comps:
                sub_comp = dict(item)
                sub_comp["name"] = f"{name} - {rc['name']}"
                sub_comp["food_area_fraction"] = round(area_frac * float(rc.get("volume_fraction", 0.5)), 4)
                sub_comp["submerged"] = True
                disaggregated.append(sub_comp)
            continue

        # Unmodified component
        disaggregated.append(item)

    return disaggregated


def density_category(name: str) -> str:
    """Classify ingredient into food density category."""
    name = name.lower()
    if any(x in name for x in ("papad", "appalam", "pappadam")):
        return "papad"
    if any(x in name for x in ("salad", "lettuce", "greens", "arugula", "spinach salad", "herb", "cilantro", "mint")):
        return "leafy_salad"
    if any(x in name for x in ("murmura", "puffed", "popcorn", "kurmura", "pori", "rice crisp", "bhel")):
        return "puffed_cereal"
    if any(x in name for x in ("mayo", "mayonnaise", "tahini", "creamy", "korma", "makhani", "butter masala", "peanut sauce")):
        return "rich_sauce"
    if any(x in name for x in ("ghee", "butter", "oil", "vanaspati", "margarine", "lard", "fat")):
        return "fat_oil"
    if any(x in name for x in ("payasam", "kheer", "halwa", "gulab jamun", "sweet")):
        return "sweets"
    if any(x in name for x in ("rice", "noodle", "pasta", "grain", "biryani", "pulao", "khichdi", "bisi bele", "quinoa", "couscous")):
        return "cooked_grain"
    if any(x in name for x in ("chicken", "beef", "pork", "lamb", "mutton", "fish", "meat", "shrimp", "prawn", "duck")):
        return "cooked_meat"
    if any(x in name for x in ("dal", "sambar", "rasam", "rajma", "chana", "channa", "chole", "pappu", "lentil", "bean", "chickpea", "curry", "stew", "salan")):
        return "legume_or_curry"
    if any(x in name for x in ("chutney", "pickle", "sauce", "gravy", "dressing")):
        return "sauce"
    if any(x in name for x in ("bread", "roti", "phulka", "chapati", "naan", "paratha", "flatbread", "puri", "bhatura", "kulcha", "tortilla")):
        return "bread"
    if any(x in name for x in ("cheese", "paneer", "yogurt", "curd", "dahi", "raita")):
        return "dairy"
    if any(x in name for x in ("fruit", "apple", "banana", "mango", "orange", "watermelon", "papaya", "muskmelon")):
        return "fruit"
    if any(x in name for x in ("vegetable", "poriyal", "fry", "sabzi", "aloo", "gobi", "onion", "pea", "carrot", "potato", "herb", "salad", "cilantro", "cabbage", "beetroot", "bhindi")):
        return "vegetable"
    return "mixed_food"


def build_weight_estimates(
    components: List[Dict[str, Any]],
    plate_scale: Dict[str, Any],
    density_table: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Convert component area shares and elevations to calibrated gram weights using dynamic physical bounding."""
    estimates = []
    total_share = sum(item.get("food_area_fraction", 0.0) for item in components)
    if total_share <= 0.0:
        total_share = 1.0
    default_density = density_table.get("mixed_food", {"min": 0.6, "max": 1.1})

    for item in components:
        name = item.get("name", "Food")
        name_lower = name.lower()
        category = density_category(name)
        density = density_table.get(category, default_density)
        elevation = item.get("elevation", "medium")
        side_used = item.get("side_view_used", False)
        area_frac = float(item.get("food_area_fraction", 0.0))

        # Ensure valid named food components always receive a non-zero area fraction
        if area_frac <= 0.0:
            area_frac = round(1.0 / max(1, len(components)), 3)

        geom_model = "extruded_prism"
        footprint_cm2 = float(plate_scale.get("food_footprint_cm2", 0.0))
        if footprint_cm2 <= 0.0:
            plate_diam = float(plate_scale.get("plate_diameter_cm", 25.0))
            footprint_cm2 = round(0.50 * np.pi * ((plate_diam / 2.0) ** 2), 1)

        normalized_share = max(0.05, area_frac) / max(0.05, total_share)
        area_cm2 = footprint_cm2 * normalized_share

        if plate_scale.get("detected"):
            # -------------------------------------------------------------
            # Dynamic Physical Geometry Modeling
            # -------------------------------------------------------------
            if category == "papad":
                # Flat sheet geometry: papad/appalam has physical thickness 0.08 - 0.18 cm
                geom_model = "flat_sheet_papad"
                t_min = 0.08
                t_max = 0.18
                t_est = 0.12
                vol_min = area_cm2 * t_min * 0.95
                vol_max = area_cm2 * t_max * 1.00
                vol_est = area_cm2 * t_est
                method = (
                    f"Physical flat-sheet model ({area_cm2:.1f} cm² × {t_est:.2f} cm thickness) "
                    f"× bulk density range [{density['min']:.2f}, {density['max']:.2f}] g/ml"
                )

            elif category == "bread":
                # Flat sheet geometry: differentiated by thin (phulka/roti) vs thick (naan/paratha)
                geom_model = "flat_sheet_bread"
                if any(x in name_lower for x in ("phulka", "roti", "chapati", "tortilla")):
                    t_min, t_max, t_est = 0.15, 0.32, 0.22
                elif any(x in name_lower for x in ("paratha", "naan", "kulcha", "bhatura", "puri")):
                    t_min, t_max, t_est = 0.30, 0.65, 0.45
                elif any(x in name_lower for x in ("bread", "toast", "bun", "slice")):
                    t_min, t_max, t_est = 0.90, 1.60, 1.25
                else:
                    t_min, t_max, t_est = 0.20, 0.45, 0.30

                vol_min = area_cm2 * t_min * 0.95
                vol_max = area_cm2 * t_max * 1.00
                vol_est = area_cm2 * t_est
                method = (
                    f"Physical flat-sheet model ({area_cm2:.1f} cm² × {t_est:.2f} cm thickness) "
                    f"× bulk density range [{density['min']:.2f}, {density['max']:.2f}] g/ml"
                )

            elif category in ("cooked_grain", "puffed_cereal") or "salad" in name_lower or elevation in ("heaped", "tall"):
                # Mounded / spherical cap geometry: food served in a dome/mound
                geom_model = "spherical_cap_mound"
                h_min = item.get("height_cm_min", 1.2)
                h_max = item.get("height_cm_max", 3.0)
                h_est = (h_min + h_max) / 2.0
                # Paraboloid / spherical cap integration factor κ ≈ 0.55 - 0.62
                vol_min = area_cm2 * h_min * 0.50
                vol_max = area_cm2 * h_max * 0.65
                vol_est = area_cm2 * h_est * 0.58
                view_desc = "side-view elevation profile" if side_used else "category height baseline"
                method = (
                    f"Calibrated mounded geometry ({area_cm2:.1f} cm² footprint, {view_desc} {h_est:.1f} cm) "
                    f"× spherical cap factor (0.58) × density range"
                )

            elif category in ("sauce", "rich_sauce", "fat_oil"):
                # Shallow layer / dollop meniscus geometry
                geom_model = "shallow_layer"
                h_min = min(float(item.get("height_cm_min", 0.3)), 0.6)
                h_max = min(float(item.get("height_cm_max", 0.8)), 1.2)
                h_est = (h_min + h_max) / 2.0
                vol_min = area_cm2 * h_min * 0.70
                vol_max = area_cm2 * h_max * 0.90
                vol_est = area_cm2 * h_est * 0.80
                method = (
                    f"Calibrated shallow layer ({area_cm2:.1f} cm² × depth {h_est:.1f} cm) "
                    f"× meniscus factor (0.80) × density range"
                )

            else:
                # Chunky / irregular pieces (curry, meat, vegetable, dairy)
                geom_model = "chunky_pack"
                h_min = float(item.get("height_cm_min", 1.0))
                h_max = float(item.get("height_cm_max", 2.2))
                h_est = (h_min + h_max) / 2.0
                shape = float(item.get("shape_factor", 0.68))
                vol_min = area_cm2 * h_min * max(0.2, shape * 0.8)
                vol_max = area_cm2 * h_max * min(1.0, shape * 1.15)
                vol_est = (vol_min + vol_max) / 2.0
                view_desc = f"side-view elevation '{elevation}'" if side_used else "standard elevation baseline"
                method = (
                    f"Calibrated footprint ({area_cm2:.1f} cm²) × {view_desc} "
                    f"× packing factor ({shape:.2f}) × density range"
                )

        else:
            # Fallback when plate calibration is unavailable (close-up / cropped photo)
            geom_model = "heuristic_fallback"
            base_vol = max(40.0, normalized_share * 420.0)
            vol_min = max(25.0, float(item.get("volume_ml_min", base_vol * 0.75)))
            vol_max = max(vol_min * 1.25, float(item.get("volume_ml_max", base_vol * 1.25)))
            vol_est = (vol_min + vol_max) / 2.0
            reason = plate_scale.get("reason", "plate rim not fully detected")
            method = f"Heuristic volume fallback (uncalibrated: {reason}) × density range"

        # Check for zero/negative volume
        if vol_max <= 0.0 or vol_est <= 0.0:
            vol_min, vol_max, vol_est = 35.0, 75.0, 55.0

        # Check if multimodal AI (Gemini with depth map & plate radius) predicted realistic weight
        ai_weight = item.get("estimated_weight_g")
        ai_vol = item.get("estimated_volume_ml")
        if ai_weight and isinstance(ai_weight, (int, float)) and ai_weight > 0:
            w_est = max(5, round(float(ai_weight) / 5) * 5)
            w_min = max(5, round(w_est * 0.85 / 5) * 5)
            w_max = max(w_min, round(w_est * 1.15 / 5) * 5)
            if ai_vol and isinstance(ai_vol, (int, float)) and ai_vol > 0:
                vol_est = float(ai_vol)
                vol_min = vol_est * 0.85
                vol_max = vol_est * 1.15
            else:
                avg_dens = (density["min"] + density["max"]) / 2.0
                vol_est = w_est / max(0.2, avg_dens)
                vol_min = vol_est * 0.85
                vol_max = vol_est * 1.15
            geom_model = "multimodal_depth_ai"
            method = (
                f"Multimodal 3D Depth & Plate Metrology AI ({w_est}g based on "
                f"{area_cm2:.1f} cm² footprint, depth map & plate radius)"
            )
            confidence = "high"
            confidence_score = 0.92
        else:
            # Dynamic physical mass bounding based on volume * density
            grams_min = max(5.0, vol_min * density["min"])
            grams_max = max(grams_min, vol_max * density["max"])
            estimate = max(5.0, vol_est * ((density["min"] + density["max"]) / 2.0))

            w_est = max(5, round(estimate / 10) * 10)
            w_min = max(5, round(grams_min / 10) * 10)
            w_max = max(w_min, round(grams_max / 10) * 10)

            # -------------------------------------------------------------
            # Confidence Indicator Synthesis
            # -------------------------------------------------------------
            confidence_score = 0.50
            if plate_scale.get("detected"):
                confidence_score += 0.25
                if plate_scale.get("confidence") == "high":
                    confidence_score += 0.10
                elif plate_scale.get("confidence") == "medium":
                    confidence_score += 0.05
                # Bonus for side-view height calibration
                if side_used:
                    confidence_score += 0.10
                # Bonus for physically constrained flat geometry
                if geom_model.startswith("flat_sheet"):
                    confidence_score += 0.05
            else:
                confidence_score = 0.30

            confidence_score = max(0.10, min(0.95, confidence_score))

            if confidence_score >= 0.75:
                confidence = "high"
            elif confidence_score >= 0.45:
                confidence = "medium"
            else:
                confidence = "low"

            # Downgrade if plate scale calibration explicitly indicated low confidence
            if not plate_scale.get("detected"):
                confidence = "low"
            elif plate_scale.get("confidence") == "low" and confidence == "high":
                confidence = "medium"

        estimates.append({
            "name": name,
            "elevation": elevation,
            "side_view_used": side_used,
            "geometry_model": geom_model,
            "volume_ml": {"min": round(vol_min), "estimate": round(vol_est), "max": round(vol_max)},
            "density_g_per_ml": {"min": density["min"], "max": density["max"]},
            "weight_g": {"min": w_min, "estimate": w_est, "max": w_max},
            "confidence": confidence,
            "confidence_score": round(confidence_score, 2),
            "method": method,
        })

    return estimates
