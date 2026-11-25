# -*- coding: utf-8 -*-
"""
ML/preprocessor.py
==================
Feature engineering, text parsing, liquid density adjustment,
count-to-weight resolution, and 3-tier preparation resolution for NutraVision.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from catboost import Pool

_UNICODE_FRACTIONS: Dict[str, float] = {
    '½': 0.5, '⅓': 1.0 / 3.0, '⅔': 2.0 / 3.0, '¼': 0.25, '¾': 0.75,
    '⅛': 0.125, '⅜': 0.375, '⅝': 0.625, '⅞': 0.875
}

_NUM_PATTERN = r'(?:(?:\d+\s+)?\d+/\d+|\d+(?:\.\d+)?|[½⅓⅔¼¾⅛⅜⅝⅞])'

_G_PER_MASS_UNIT: Dict[str, float] = {
    "g": 1.0, "gm": 1.0, "gms": 1.0, "gram": 1.0, "grams": 1.0,
    "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.59, "lbs": 453.59, "pound": 453.59, "pounds": 453.59,
}

DEVANAGARI_FOOD_MAP: Dict[str, str] = {
    "रोटी": "roti", "रोटियां": "roti", "रोटियाँ": "roti",
    "फुल्का": "phulka", "चपाती": "chapati", "पराठा": "paratha", "पराठे": "paratha",
    "पूरी": "puri", "चावल": "rice", "भात": "rice", "दाल": "dal",
    "सांभर": "sambar", "रसम": "rasam", "पनीर": "paneer", "चिकन": "chicken",
    "मुर्गा": "chicken", "अंडा": "egg", "अंडे": "egg", "अण्डा": "egg",
    "मटन": "mutton", "गोश्त": "meat", "मछली": "fish", "दही": "curd",
    "रायता": "raita", "दूध": "milk", "मक्खन": "butter", "घी": "ghee",
    "सलाद": "salad", "सब्जी": "vegetables", "सब्जियां": "vegetables",
    "आलू": "potato", "गोभी": "cauliflower", "प्याज": "onion", "टमाटर": "tomato",
    "पालक": "spinach", "भिंडी": "bhindi", "चना": "chana", "छोले": "chole",
    "राजमा": "rajma", "खीर": "kheer", "हलवा": "halwa", "गुलाब जामुन": "gulab jamun",
    "समोसा": "samosa", "डोसा": "dosa", "इडली": "idli", "बिरयानी": "biryani",
    "पोहा": "poha", "उपमा": "upma", "पापड़": "papad", "चटनी": "chutney",
    "अचार": "pickle", "पानी": "water", "चाय": "tea", "कॉफी": "coffee",
    "ग्राम": "g", "किलो": "kg", "चम्मच": "tbsp", "कप": "cup",
}


def transliterate_devanagari(text: str) -> str:
    """Normalize Devanagari Hindi food words and units to English equivalents."""
    if not text:
        return ""
    result = text
    for dev, eng in DEVANAGARI_FOOD_MAP.items():
        result = result.replace(dev, f" {eng} ")
    return re.sub(r"\s+", " ", result).strip()

_ML_PER_VOL_UNIT: Dict[str, float] = {
    "ml": 1.0, "millilitre": 1.0, "millilitres": 1.0, "milliliter": 1.0, "milliliters": 1.0,
    "l": 1000.0, "liter": 1000.0, "liters": 1000.0, "litre": 1000.0, "litres": 1000.0,
    "cup": 240.0, "cups": 240.0,
    "tbsp": 15.0, "tablespoon": 15.0, "tablespoons": 15.0,
    "tsp": 5.0, "teaspoon": 5.0, "teaspoons": 5.0,
}

# Specific gravity / densities for common liquids (g/ml)
LIQUID_DENSITIES: Dict[str, float] = {
    "oil": 0.92,
    "olive oil": 0.92,
    "coconut oil": 0.92,
    "vegetable oil": 0.92,
    "mustard oil": 0.92,
    "sesame oil": 0.92,
    "canola oil": 0.92,
    "sunflower oil": 0.92,
    "ghee": 0.92,
    "clarified butter": 0.92,
    "honey": 1.42,
    "syrup": 1.35,
    "maple syrup": 1.33,
    "milk": 1.03,
    "buttermilk": 1.03,
    "cream": 1.01,
    "yogurt": 1.04,
    "curd": 1.04,
    "water": 1.00,
    "juice": 1.04,
    "vinegar": 1.01,
    "sauce": 1.10,
    "soy sauce": 1.15,
}

# Standard realistic weights (in grams) for common countable portion items
COUNT_PORTION_WEIGHTS: Dict[str, float] = {
    # Breads & Grains
    "roti": 35.0, "rotis": 35.0,
    "chapati": 35.0, "chapatis": 35.0, "chapatti": 35.0, "chapattis": 35.0,
    "phulka": 30.0, "phulkas": 30.0,
    "paratha": 80.0, "parathas": 80.0,
    "naan": 100.0, "naans": 100.0,
    "puri": 30.0, "puris": 30.0, "poori": 30.0, "pooris": 30.0,
    "bhatura": 80.0, "bhature": 80.0,
    "dosa": 90.0, "dosas": 90.0,
    "idli": 50.0, "idlis": 50.0,
    "vada": 60.0, "vadas": 60.0,
    "samosa": 75.0, "samosas": 75.0,
    "bread": 35.0, "slice": 35.0, "slices": 35.0, "toast": 35.0,
    "bun": 60.0, "buns": 60.0,
    "tortilla": 45.0, "tortillas": 45.0,

    # Eggs & Proteins
    "egg": 50.0, "eggs": 50.0,
    "boiled egg": 50.0, "boiled eggs": 50.0,
    "fried egg": 55.0, "fried eggs": 55.0,
    "omelette": 100.0, "omelet": 100.0,
    "patty": 85.0, "patties": 85.0,
    "sausage": 50.0, "sausages": 50.0,

    # Fruits
    "apple": 150.0, "apples": 150.0,
    "banana": 120.0, "bananas": 120.0,
    "orange": 130.0, "oranges": 130.0,
    "mango": 200.0, "mangoes": 200.0,
    "lemon": 60.0, "lemons": 60.0,
    "lime": 50.0, "limes": 50.0,
    "peach": 150.0, "peaches": 150.0,
    "pear": 160.0, "pears": 160.0,
    "guava": 100.0, "guavas": 100.0,

    # Vegetables & Snacks
    "potato": 150.0, "potatoes": 150.0,
    "tomato": 100.0, "tomatoes": 100.0,
    "onion": 100.0, "onions": 100.0,
    "carrot": 70.0, "carrots": 70.0,
    "cucumber": 200.0, "cucumbers": 200.0,
    "biscuit": 15.0, "biscuits": 15.0,
    "cookie": 20.0, "cookies": 20.0,
    "papad": 15.0, "papads": 15.0,
}


def parse_fraction_or_float(val_str: str) -> float:
    """Parse integer, float, unicode fraction, or common fraction string like '1/2' or '1 1/2'."""
    val_str = val_str.strip()
    for uf, val in _UNICODE_FRACTIONS.items():
        if uf in val_str:
            base_str = val_str.replace(uf, "").strip()
            base = float(base_str) if base_str else 0.0
            return base + val
    if "/" in val_str:
        parts = val_str.split()
        if len(parts) == 2:
            base = float(parts[0])
            num, denom = parts[1].split("/")
            return base + float(num) / float(denom)
        elif len(parts) == 1:
            num, denom = parts[0].split("/")
            return float(num) / float(denom)
    return float(val_str)


def parse_weight(text: str) -> Tuple[float, str]:
    """
    Return (weight_g, text_without_weight_token).
    Supports:
      - Mass units: g, kg, oz, lb (e.g. '150g', '0.5 kg', '1/4 kg chicken')
      - Volume units with specific gravity: ml, l, cup, tbsp, tsp (e.g. '100ml olive oil', '3/4 cup milk', '1 tbsp honey')
      - Count portions: '2 rotis', '2 eggs', '1 idli', '1/2 apple'
      - Standalone count foods: 'apple' -> 150g, 'egg' -> 50g
      - Defaults to 100g if omitted and no count food matches.
    """
    text_clean = transliterate_devanagari(text.strip())

    # 1. Check explicit mass units (e.g. 150g, 1/4 kg, 2.5 lbs)
    mass_units_pattern = "|".join(sorted(_G_PER_MASS_UNIT.keys(), key=lambda x: -len(x)))
    m_mass = re.search(r"(" + _NUM_PATTERN + r")\s*(" + mass_units_pattern + r")\b", text_clean, re.IGNORECASE)
    if m_mass:
        try:
            qty = parse_fraction_or_float(m_mass.group(1))
            unit = m_mass.group(2).lower()
            factor = _G_PER_MASS_UNIT.get(unit, 1.0)
            weight_g = qty * factor
            cleaned = (text_clean[:m_mass.start()] + text_clean[m_mass.end():]).strip(" ,")
            return round(weight_g, 2), cleaned or text_clean
        except (ValueError, ZeroDivisionError):
            pass

    # 2. Check volume units with specific gravity (e.g. 100ml oil, 3/4 cup milk, 1 tbsp honey)
    vol_units_pattern = "|".join(sorted(_ML_PER_VOL_UNIT.keys(), key=lambda x: -len(x)))
    m_vol = re.search(r"(" + _NUM_PATTERN + r")\s*(" + vol_units_pattern + r")\b", text_clean, re.IGNORECASE)
    if m_vol:
        try:
            qty = parse_fraction_or_float(m_vol.group(1))
            unit = m_vol.group(2).lower()
            vol_ml = qty * _ML_PER_VOL_UNIT.get(unit, 1.0)
            cleaned = (text_clean[:m_vol.start()] + text_clean[m_vol.end():]).strip(" ,")
            # Determine density based on cleaned food name
            density = 1.0
            cleaned_l = cleaned.lower()
            for liq, d in sorted(LIQUID_DENSITIES.items(), key=lambda x: -len(x[0])):
                if re.search(r"\b" + re.escape(liq) + r"\b", cleaned_l):
                    density = d
                    break
            weight_g = vol_ml * density
            return round(weight_g, 2), cleaned or text_clean
        except (ValueError, ZeroDivisionError):
            pass

    # 3. Check count portions: e.g. '2 rotis', '1 idli', '2 eggs', '1/2 apple'
    m_count = re.search(r"^\s*(" + _NUM_PATTERN + r")\s+(.*)", text_clean, re.IGNORECASE)
    if m_count:
        try:
            qty = parse_fraction_or_float(m_count.group(1))
            remainder = m_count.group(2).strip(" ,")
            rem_l = remainder.lower()
            for item, unit_wt in sorted(COUNT_PORTION_WEIGHTS.items(), key=lambda x: -len(x[0])):
                pattern = r"\b" + re.escape(item) + (r"\b" if item.endswith("s") else r"(?:s|es)?\b")
                if re.search(pattern, rem_l):
                    return round(qty * unit_wt, 2), remainder
            # Fallback for count with unlisted item: qty * 100.0g
            return round(qty * 100.0, 2), remainder
        except (ValueError, ZeroDivisionError):
            pass

    # 4. Check bare count item: e.g. 'egg', 'apple', 'idli', 'roti'
    text_l = text_clean.lower()
    for item, unit_wt in sorted(COUNT_PORTION_WEIGHTS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(item) + (r"\b" if item.endswith("s") else r"(?:s|es)?\b")
        if re.search(pattern, text_l):
            return round(unit_wt, 2), text_clean

    return 100.0, text_clean


def detect_preparation(text: str, prep_keywords: Dict[str, List[str]]) -> str:
    """Detect preparation method using word boundaries to prevent substring collisions."""
    text_l = transliterate_devanagari(text).lower()
    for prep, keywords in prep_keywords.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + (r"\b" if kw.endswith("s") else r"(?:s|es)?\b")
            if re.search(pattern, text_l):
                return prep
    return "unknown"


def detect_category(text: str, category_keywords: Any) -> str:
    """Detect food category using word boundaries to prevent substring collisions."""
    text_l = transliterate_devanagari(text).lower()
    if isinstance(category_keywords, dict):
        for cat, kws in category_keywords.items():
            for kw in kws:
                pattern = r"\b" + re.escape(kw) + (r"\b" if kw.endswith("s") else r"(?:s|es)?\b")
                if re.search(pattern, text_l):
                    return cat
    elif isinstance(category_keywords, (list, set)):
        for cat in category_keywords:
            pattern = r"\b" + re.escape(cat) + (r"\b" if cat.endswith("s") else r"(?:s|es)?\b")
            if re.search(pattern, text_l):
                return cat
    return "unknown"


def detect_attributes(text: str, attr_names: List[str], attr_keywords: Dict[str, List[str]]) -> List[int]:
    """Detect binary attributes (e.g. skinless, boneless) using word boundary regex."""
    text_l = text.lower()
    flags = []
    for attr in attr_names:
        kws = attr_keywords.get(attr, [])
        matched = 0
        for kw in kws:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_l):
                matched = 1
                break
        flags.append(matched)
    return flags


# Contextual preparation mappings
DISH_PREP_CONTEXT: Dict[str, Dict[str, str]] = {
    # Indian Rice & Biryanis
    "biryani":      {"rice": "boiled", "chicken": "roasted", "mutton": "roasted",
                     "lamb": "roasted", "egg": "boiled", "onion": "fried",
                     "potato": "boiled"},
    "pulao":        {"rice": "boiled", "chicken": "boiled", "vegetable": "boiled",
                     "pea": "boiled", "paneer": "fried"},
    "khichdi":      {"rice": "boiled", "dal": "boiled", "lentil": "boiled", "moong": "boiled"},
    "bisi bele":    {"rice": "boiled", "dal": "boiled", "vegetable": "boiled"},
    "fried rice":   {"rice": "boiled", "egg": "fried", "chicken": "fried",
                     "vegetable": "fried", "onion": "fried"},

    # Indian Thalis & Combos
    "thali":        {"rice": "boiled", "phulka": "baked", "roti": "baked", "dal": "boiled",
                     "curd": "raw", "sambar": "boiled", "papad": "roasted", "salad": "raw"},

    # Indian Dals & Soups
    "dal":          {"lentil": "boiled", "dal": "boiled", "onion": "fried", "tomato": "boiled"},
    "dal makhani":  {"lentil": "boiled", "dal": "boiled", "rajma": "boiled", "butter": "raw", "cream": "raw"},
    "sambar":       {"dal": "boiled", "lentil": "boiled", "vegetable": "boiled", "drumstick": "boiled"},
    "rasam":        {"tomato": "boiled", "pepper": "boiled", "tamarind": "boiled"},
    "pappu":        {"dal": "boiled", "toor": "boiled", "tomato": "boiled", "spinach": "boiled"},

    # Indian Curries & Gravies
    "curry":        {"chicken": "boiled", "mutton": "boiled", "lamb": "boiled",
                     "paneer": "boiled", "potato": "boiled", "rice": "boiled", "fish": "boiled"},
    "palak paneer": {"spinach": "boiled", "paneer": "boiled", "cream": "raw"},
    "butter chicken":{"chicken": "roasted", "butter": "raw", "cream": "raw", "tomato": "boiled"},
    "tikka masala": {"chicken": "roasted", "paneer": "roasted", "cream": "raw"},
    "rajma":        {"rajma": "boiled", "bean": "boiled", "onion": "fried", "tomato": "boiled"},
    "chole":        {"chickpea": "boiled", "chana": "boiled", "onion": "fried"},
    "chana masala": {"chickpea": "boiled", "chana": "boiled", "onion": "fried"},
    "kofta":        {"kofta": "fried", "potato": "boiled", "paneer": "fried"},

    # Indian Dry Sabzis & Fries
    "aloo fry":     {"potato": "fried", "aloo": "fried"},
    "aloo gobi":    {"potato": "fried", "cauliflower": "fried", "gobi": "fried"},
    "poriyal":      {"beetroot": "fried", "cabbage": "fried", "beans": "fried", "coconut": "raw"},
    "fry":          {"potato": "fried", "dondakaya": "fried", "kakarakaya": "fried", "cabbage": "fried"},

    # Indian Breads
    "paratha":      {"flour": "fried", "potato": "boiled", "paneer": "raw", "gobi": "boiled"},
    "roti":         {"wheat": "baked", "flour": "baked", "ghee": "raw"},
    "phulka":       {"wheat": "baked", "flour": "baked"},
    "naan":         {"flour": "baked", "butter": "raw", "garlic": "baked"},

    # South Indian Tiffin
    "dosa":         {"potato": "boiled", "onion": "fried", "crepe": "fried"},
    "idli":         {"rice": "steamed", "dal": "steamed"},

    # Western & General
    "salad":        {"chicken": "grilled", "egg": "boiled", "tuna": "raw",
                     "vegetable": "raw", "lettuce": "raw", "tomato": "raw"},
    "sandwich":     {"chicken": "grilled", "egg": "boiled", "vegetable": "raw"},
    "burger":       {"chicken": "fried", "beef": "fried", "patty": "fried"},
    "stir fry":     {"chicken": "fried", "beef": "fried", "vegetable": "fried",
                     "noodle": "boiled", "rice": "boiled"},
    "soup":         {"chicken": "boiled", "vegetable": "boiled", "potato": "boiled"},
    "pasta":        {"pasta": "boiled", "chicken": "grilled", "vegetable": "boiled"},
    "pizza":        {"chicken": "roasted", "cheese": "raw", "vegetable": "roasted"},
}

INGREDIENT_DEFAULT_PREP: Dict[str, str] = {
    "rice": "boiled",    "basmati": "boiled", "noodle": "boiled",
    "pasta": "boiled",   "quinoa": "boiled",  "oat": "boiled",
    "bread": "baked",    "roti": "baked",     "naan": "baked",
    "chapati": "baked",  "paratha": "fried",  "phulka": "baked",
    "puri": "fried",     "bhatura": "fried",  "dosa": "fried",
    "idli": "steamed",   "upma": "boiled",    "poha": "fried",
    "khichdi": "boiled", "papad": "roasted",
    "chicken": "boiled", "mutton": "boiled",  "lamb": "boiled",
    "beef": "boiled",    "pork": "boiled",    "fish": "boiled",
    "prawn": "boiled",   "shrimp": "boiled",  "egg": "boiled",
    "tuna": "raw",       "salmon": "raw",
    "dal": "boiled",     "lentil": "boiled",  "bean": "boiled",
    "chickpea": "boiled","pea": "boiled",     "sambar": "boiled",
    "rasam": "boiled",   "rajma": "boiled",   "chana": "boiled",
    "chole": "boiled",   "toor": "boiled",    "moong": "boiled",
    "urad": "boiled",    "masoor": "boiled",
    "potato": "boiled",  "carrot": "boiled",  "spinach": "boiled",
    "onion": "boiled",   "tomato": "boiled",  "broccoli": "boiled",
    "cauliflower": "boiled", "cabbage": "boiled", "corn": "boiled",
    "aloo": "boiled",    "gobi": "boiled",    "bhindi": "fried",
    "baingan": "roasted","poriyal": "fried",  "eggplant": "roasted",
    "brinjal": "roasted",
    "apple": "raw",   "banana": "raw",  "mango": "raw",
    "orange": "raw",  "grape": "raw",   "berry": "raw",
    "watermelon": "raw", "papaya": "raw", "muskmelon": "raw",
    "milk": "raw",    "cheese": "raw",  "paneer": "raw",
    "yogurt": "raw",  "curd": "raw",    "dahi": "raw",
    "raita": "raw",   "butter": "raw",  "ghee": "raw",
    "chutney": "raw", "pickle": "raw",
    "peanut": "raw",  "nut": "raw",     "almond": "raw",
    "walnut": "raw",  "cashew": "raw",  "pistachio": "raw",
    "goat": "roasted",
}


def resolve_preparation(dish: Optional[str], food_name: str, prep_keywords: Dict[str, List[str]]) -> str:
    """3-Tier Preparation Resolution: Explicit keyword > Dish context > Ingredient smart default."""
    prep = detect_preparation(food_name, prep_keywords)
    if prep != "unknown":
        return prep
    if dish:
        dish_l = dish.lower()
        food_l = food_name.lower()
        for dish_key, ingr_map in DISH_PREP_CONTEXT.items():
            dish_pattern = r"\b" + re.escape(dish_key) + r"\b"
            if re.search(dish_pattern, dish_l) or dish_key in dish_l:
                for ingr_key, d_prep in ingr_map.items():
                    pattern = r"\b" + re.escape(ingr_key) + (r"\b" if ingr_key.endswith("s") else r"(?:s|es)?\b")
                    if re.search(pattern, food_l):
                        return d_prep
    food_l = food_name.lower()
    for ingr_key, default_prep in INGREDIENT_DEFAULT_PREP.items():
        pattern = r"\b" + re.escape(ingr_key) + (r"\b" if ingr_key.endswith("s") else r"(?:s|es)?\b")
        if re.search(pattern, food_l):
            return default_prep
    return "boiled"


def build_catboost_batch_pool(
    food_names: List[str],
    categories: List[str],
    preparations: List[str],
    attributes_list: List[List[int]],
    vectorizer: Any,
    svd: Any,
    attr_cols: List[str],
    cat_features: List[str],
) -> Pool:
    """Build a single CatBoost Pool for a batch of ingredients matching the model's feature schema."""
    clean_food_names = [transliterate_devanagari(n) for n in food_names]
    tfidf_mat = vectorizer.transform(clean_food_names)
    svd_mat   = svd.transform(tfidf_mat)  # shape (N, 64)
    text_df = pd.DataFrame(
        svd_mat,
        columns=[f"text_{i}" for i in range(svd_mat.shape[1])]
    )
    cat_df = pd.DataFrame({
        "category": [str(c) for c in categories],
        "preparation": [str(p) for p in preparations],
    })
    attr_df = pd.DataFrame(
        attributes_list,
        columns=attr_cols
    )
    df = pd.concat([text_df, cat_df, attr_df], axis=1)
    return Pool(df, cat_features=cat_features)


def build_catboost_pool(
    food_name: str,
    category: str,
    preparation: str,
    attributes: List[int],
    vectorizer: Any,
    svd: Any,
    attr_cols: List[str],
    cat_features: List[str],
) -> Pool:
    """Build a CatBoost Pool matching the model's feature schema for a single item."""
    return build_catboost_batch_pool(
        food_names=[food_name],
        categories=[category],
        preparations=[preparation],
        attributes_list=[attributes],
        vectorizer=vectorizer,
        svd=svd,
        attr_cols=attr_cols,
        cat_features=cat_features,
    )
