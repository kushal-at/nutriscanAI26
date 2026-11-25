# -*- coding: utf-8 -*-
"""
ML/predictor.py
===============
Model registry, artifact loader, vectorized CatBoost prediction engine,
Physical Mass Conservation, and Thermodynamic Atwater Reconciliation for NutraVision.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import joblib
from catboost import CatBoostRegressor

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
if str(_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_DIR.parent))

try:
    from Intelligence.ML.schemas import (
        IngredientIn, NutrientResult, RecipeTotals, PredictResponse,
        VisionNutrientResult, VisionPredictResponse
    )
    from Intelligence.ML.preprocessor import (
        parse_weight, detect_category, detect_attributes, resolve_preparation,
        build_catboost_batch_pool
    )
except ModuleNotFoundError:
    try:
        from schemas import (
            IngredientIn, NutrientResult, RecipeTotals, PredictResponse,
            VisionNutrientResult, VisionPredictResponse
        )
        from preprocessor import (
            parse_weight, detect_category, detect_attributes, resolve_preparation,
            build_catboost_batch_pool
        )
    except ModuleNotFoundError:
        from ML.schemas import (
            IngredientIn, NutrientResult, RecipeTotals, PredictResponse,
            VisionNutrientResult, VisionPredictResponse
        )
        from ML.preprocessor import (
            parse_weight, detect_category, detect_attributes, resolve_preparation,
            build_catboost_batch_pool
        )

TARGETS = ["energy_kcal", "carb_g", "protein_g", "fat_g", "fibre_g"]
CAT_FEATURES = ["category", "preparation"]


CANONICAL_ANCHORS: Dict[str, Dict[str, float]] = {
    # Zero-calorie beverages and non-caloric items
    "water": {"energy_kcal": 0.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fibre_g": 0.0},
    "plain water": {"energy_kcal": 0.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fibre_g": 0.0},
    "drinking water": {"energy_kcal": 0.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fibre_g": 0.0},
    "sparkling water": {"energy_kcal": 0.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fibre_g": 0.0},
    "black coffee": {"energy_kcal": 2.0, "carb_g": 0.0, "protein_g": 0.3, "fat_g": 0.0, "fibre_g": 0.0},
    "green tea": {"energy_kcal": 1.0, "carb_g": 0.0, "protein_g": 0.2, "fat_g": 0.0, "fibre_g": 0.0},
    "black tea": {"energy_kcal": 1.0, "carb_g": 0.0, "protein_g": 0.2, "fat_g": 0.0, "fibre_g": 0.0},
    "diet coke": {"energy_kcal": 0.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fibre_g": 0.0},
    "diet soda": {"energy_kcal": 0.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fibre_g": 0.0},
    "salt": {"energy_kcal": 0.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fibre_g": 0.0},
    # Pure fats and culinary oils (per 100g)
    "ghee": {"energy_kcal": 900.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "desi ghee": {"energy_kcal": 900.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "clarified butter": {"energy_kcal": 900.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "oil": {"energy_kcal": 884.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "cooking oil": {"energy_kcal": 884.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "olive oil": {"energy_kcal": 884.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "coconut oil": {"energy_kcal": 862.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "mustard oil": {"energy_kcal": 884.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "sunflower oil": {"energy_kcal": 884.0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 100.0, "fibre_g": 0.0},
    "butter": {"energy_kcal": 717.0, "carb_g": 0.1, "protein_g": 0.9, "fat_g": 81.1, "fibre_g": 0.0},
}


class PredictionError(Exception):
    """Domain exception raised when feature preprocessing or model inference fails."""
    pass


class NutritionPredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self.svd = joblib.load(model_dir / "svd.joblib")
        self.attribute_keywords = joblib.load(model_dir / "attribute_keywords.joblib")
        self.attribute_names = joblib.load(model_dir / "attribute_names.joblib")
        self.category_keywords = joblib.load(model_dir / "category_keywords.joblib")
        self.preparation_keywords = joblib.load(model_dir / "preparation_keywords.joblib")
        self.metadata = joblib.load(model_dir / "metadata.joblib")

        self.attr_cols = [f"attr_{a}" for a in self.attribute_names]

        self.catboost_models: Dict[str, CatBoostRegressor] = {}
        for target in TARGETS:
            cb = CatBoostRegressor()
            cb.load_model(str(model_dir / f"{target}.cbm"))
            self.catboost_models[target] = cb

    def _predict_batch_raw(
        self,
        food_names: List[str],
        categories: List[str],
        preparations: List[str],
        attributes_list: List[List[int]],
    ) -> Dict[str, np.ndarray]:
        """
        Builds a single batch Pool and performs 5 model predictions (one per target)
        across all N items simultaneously.
        """
        if not food_names:
            return {t: np.array([], dtype=float) for t in TARGETS}

        try:
            pool = build_catboost_batch_pool(
                food_names=food_names,
                categories=categories,
                preparations=preparations,
                attributes_list=attributes_list,
                vectorizer=self.vectorizer,
                svd=self.svd,
                attr_cols=self.attr_cols,
                cat_features=CAT_FEATURES,
            )
        except Exception as exc:
            raise PredictionError(f"Vectorized feature extraction failed: {exc}") from exc

        preds: Dict[str, np.ndarray] = {}
        for target, model in self.catboost_models.items():
            try:
                raw_val = model.predict(pool)
                raw_val = np.asarray(raw_val, dtype=float).ravel()
                preds[target] = np.maximum(raw_val, 0.0)
            except Exception as exc:
                raise PredictionError(f"CatBoost inference error for '{target}': {exc}") from exc

        return preds

    @staticmethod
    def _reconcile_and_scale(
        raw_preds: Dict[str, np.ndarray],
        weights: List[float],
        food_names: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Applies:
        1. Canonical Physical Ground-Truth Anchors (e.g. 0 kcal for pure water/diet drinks; pure fat for ghee/oils).
        2. Physical Mass Conservation: sum of macros (protein, carb, fat, fibre) <= 100g per 100g.
        3. Thermodynamic Atwater Reconciliation:
           kcal_atwater = 4 * protein + 4 * carb + 9 * fat + 2 * fibre.
           Energy is blended (50% raw, 50% atwater) and constrained to within ±10% of kcal_atwater.
        4. Scaling by individual portion weight (scaled = per_100g * weight_g / 100.0).
        """
        reconciled: List[Dict[str, float]] = []
        n_items = len(weights)

        for i in range(n_items):
            weight_g = weights[i]
            food_name = (food_names[i] if food_names and i < len(food_names) else "").strip().lower()

            # 1. Canonical Physical Anchor Overrides
            matched_anchor = None
            if food_name in CANONICAL_ANCHORS:
                matched_anchor = CANONICAL_ANCHORS[food_name]
            else:
                for anchor_key, anchor_vals in CANONICAL_ANCHORS.items():
                    if re.search(r"\b" + re.escape(anchor_key) + r"\b", food_name):
                        matched_anchor = anchor_vals
                        break

            if matched_anchor is not None:
                scale = weight_g / 100.0
                reconciled.append({
                    "energy_kcal": round(matched_anchor["energy_kcal"] * scale, 2),
                    "carb_g":      round(matched_anchor["carb_g"] * scale, 2),
                    "protein_g":   round(matched_anchor["protein_g"] * scale, 2),
                    "fat_g":       round(matched_anchor["fat_g"] * scale, 2),
                    "fibre_g":     round(matched_anchor["fibre_g"] * scale, 2),
                })
                continue

            carb_g = float(raw_preds["carb_g"][i])
            protein_g = float(raw_preds["protein_g"][i])
            fat_g = float(raw_preds["fat_g"][i])
            fibre_g = float(raw_preds["fibre_g"][i])
            raw_kcal = float(raw_preds["energy_kcal"][i])

            # 2. Physical Mass Conservation per 100g
            macro_sum = protein_g + carb_g + fat_g + fibre_g
            if macro_sum > 100.0:
                factor = 100.0 / macro_sum
                protein_g *= factor
                carb_g *= factor
                fat_g *= factor
                fibre_g *= factor

            # 3. Thermodynamic Atwater Reconciliation per 100g
            # Standard Atwater factors: Protein: 4, Carb: 4, Fat: 9, Dietary Fibre: 2 kcal/g
            kcal_atwater = 4.0 * protein_g + 4.0 * carb_g + 9.0 * fat_g + 2.0 * fibre_g
            if kcal_atwater > 0.0:
                blended_kcal = 0.5 * raw_kcal + 0.5 * kcal_atwater
                # Constrain energy to within ±10% of thermodynamic Atwater value
                min_kcal = 0.9 * kcal_atwater
                max_kcal = 1.1 * kcal_atwater
                energy_kcal = max(min_kcal, min(max_kcal, blended_kcal))
            else:
                energy_kcal = 0.0

            # 4. Scale by portion weight
            scale = weight_g / 100.0
            reconciled.append({
                "energy_kcal": round(energy_kcal * scale, 2),
                "carb_g":      round(carb_g * scale, 2),
                "protein_g":   round(protein_g * scale, 2),
                "fat_g":       round(fat_g * scale, 2),
                "fibre_g":     round(fibre_g * scale, 2),
            })

        return reconciled

    def predict_ingredients(self, ingredients: List[IngredientIn]) -> PredictResponse:
        if not ingredients:
            return PredictResponse(
                ingredients=[],
                recipe_totals=RecipeTotals(
                    total_weight_g=0.0,
                    energy_kcal=0.0,
                    carb_g=0.0,
                    protein_g=0.0,
                    fat_g=0.0,
                    fibre_g=0.0,
                ),
            )

        food_names: List[str] = []
        weights: List[float] = []
        categories: List[str] = []
        preparations: List[str] = []
        attributes_list: List[List[int]] = []
        meta_items: List[Dict[str, Any]] = []

        for item in ingredients:
            parsed_w, food_name = parse_weight(item.name)
            weight_g = item.weight_g if item.weight_g is not None else parsed_w
            if weight_g < 0:
                raise ValueError(f"Weight cannot be negative for '{item.name}': {weight_g}")

            preparation = item.preparation or resolve_preparation(None, food_name, self.preparation_keywords)
            category    = item.category    or detect_category(item.name, self.category_keywords)
            attributes  = detect_attributes(item.name, self.attribute_names, self.attribute_keywords)

            food_names.append(food_name)
            weights.append(weight_g)
            categories.append(category)
            preparations.append(preparation)
            attributes_list.append(attributes)
            meta_items.append({
                "entered_item": item.name,
                "parsed_food": food_name,
                "weight_g": weight_g,
                "category": category,
                "preparation": preparation,
            })

        # Single batch prediction across all targets (5 model calls total)
        raw_preds = self._predict_batch_raw(food_names, categories, preparations, attributes_list)
        scaled_nutrients = self._reconcile_and_scale(raw_preds, weights, food_names)

        results: List[NutrientResult] = []
        totals = {t: 0.0 for t in TARGETS}
        total_weight = 0.0

        for meta, nutr in zip(meta_items, scaled_nutrients):
            results.append(NutrientResult(
                entered_item=meta["entered_item"],
                parsed_food=meta["parsed_food"],
                weight_g=meta["weight_g"],
                category=meta["category"],
                preparation=meta["preparation"],
                **nutr,
            ))
            total_weight += meta["weight_g"]
            for t in TARGETS:
                totals[t] += nutr[t]

        return PredictResponse(
            ingredients=results,
            recipe_totals=RecipeTotals(
                total_weight_g=round(total_weight, 2),
                **{t: round(v, 2) for t, v in totals.items()},
            ),
        )

    def predict_vision_pairs(self, dish: Optional[str], pairs: List[Tuple[str, float]]) -> VisionPredictResponse:
        if not pairs:
            return VisionPredictResponse(
                dish=dish,
                nutrition=[],
                recipe_totals=RecipeTotals(
                    total_weight_g=0.0,
                    energy_kcal=0.0,
                    carb_g=0.0,
                    protein_g=0.0,
                    fat_g=0.0,
                    fibre_g=0.0,
                ),
            )

        food_names: List[str] = []
        weights: List[float] = []
        categories: List[str] = []
        preparations: List[str] = []
        attributes_list: List[List[int]] = []

        for food_name, weight_g in pairs:
            if weight_g < 0:
                raise ValueError(f"Weight cannot be negative for '{food_name}': {weight_g}")

            preparation = resolve_preparation(dish, food_name, self.preparation_keywords)
            category    = detect_category(food_name, self.category_keywords)
            attributes  = detect_attributes(food_name, self.attribute_names, self.attribute_keywords)

            food_names.append(food_name)
            weights.append(weight_g)
            categories.append(category)
            preparations.append(preparation)
            attributes_list.append(attributes)

        # Single batch prediction across all targets (5 model calls total)
        raw_preds = self._predict_batch_raw(food_names, categories, preparations, attributes_list)
        scaled_nutrients = self._reconcile_and_scale(raw_preds, weights, food_names)

        results: List[VisionNutrientResult] = []
        totals = {t: 0.0 for t in TARGETS}
        total_weight = 0.0

        for food_name, weight_g, nutr in zip(food_names, weights, scaled_nutrients):
            results.append(VisionNutrientResult(
                ingredient=food_name,
                weight_g=weight_g,
                **nutr,
            ))
            total_weight += weight_g
            for t in TARGETS:
                totals[t] += nutr[t]

        return VisionPredictResponse(
            dish=dish,
            nutrition=results,
            recipe_totals=RecipeTotals(
                total_weight_g=round(total_weight, 2),
                **{t: round(v, 2) for t, v in totals.items()},
            ),
        )
