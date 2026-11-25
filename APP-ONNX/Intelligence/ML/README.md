---
title: NutriScan ML Nutrition API
emoji: 🥗
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: 5.12.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# NutriScan ML — CatBoost Nutrition Regression API

Multi-target CatBoost regression service predicting `energy_kcal`, `carb_g`, `protein_g`, `fat_g`, and `fibre_g` with Atwater factor reconciliation.

### API Endpoints
- `GET /health` — Service & model status
- `POST /predict` — Predict nutrition from ingredient list
- `POST /predict/vision` — Predict nutrition from vision portion estimates
- `GET /docs` — Interactive Swagger UI
