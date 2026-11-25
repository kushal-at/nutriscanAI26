---
title: NutriScan DL Vision API
emoji: 📸
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.12.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# NutriScan DL — Dual-View Food Vision & Portion Estimator API

DeepLabV3+ semantic segmentation, Hough circle plate metrology, Gemini VLM reasoning, and volume-to-gram portion estimation.

### Environment Secrets (Set in Space Settings)
- `GEMINI_API_KEYS`: Comma-separated Google Gemini API keys for VLM recognition.
- `GEMINI_MODEL` (optional): Defaults to `gemini-3-flash-preview`.

### API Endpoints
- `GET /health` — Service, model, and Gemini router status
- `POST /vision` — Analyze top-down & side-view meal photos
- `GET /gemini/keys` — Key pool status
- `GET /docs` — Interactive Swagger UI
