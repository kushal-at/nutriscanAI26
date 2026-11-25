# 🥗 NutraVision — Dual-View AI Nutrition Analyzer & Food Weight Estimator

NutraVision is an offline-capable, dual-microservice AI platform that estimates food portion weights and complete nutritional profiles from paired top-down and side-view meal photographs.

The project combines **PyTorch DeepLabV3+ semantic segmentation**, local **Qwen3-VL (2B)** multimodal reasoning, **calibrated plate geometry**, and **CatBoost multi-target regression** enriched with authentic Indian food datasets from **CVIT IIIT Hyderabad (ICVGIP 2025)** and **ICMR-NIN**.

---

## 🏗️ 3-Tier Multi-Port Microservice Architecture

```
                      [ User Photos: Top-Down + Side-View ]
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   FrontEnd & BackEnd (Port 8000)  │
                     ├───────────────────────────────────┤
                     │ 1. Serves Web UI & Static Assets  │
                     │ 2. Dual-View Analysis Gateway     │
                     │ 3. NutriBot Conversational Agent  │
                     └─────────────┬───────────────┬─────┘
                                   │               │
             POST /vision (images) │               │ POST /predict/vision (weights)
                                   ▼               ▼
        ┌───────────────────────────────────┐    ┌───────────────────────────────────┐
        │  Deep Learning: DL (Port 8001)    │    │  Machine Learning: ML (Port 8002) │
        ├───────────────────────────────────┤    ├───────────────────────────────────┤
        │ 1. DeepLabV3+ MobileNetV2 (104cls)│    │ 1. 3-Tier Preparation Resolution  │
        │ 2. Hough Circle Plate Calibration │    │ 2. Char-wb TF-IDF (2-5) + 64D SVD │
        │ 3. Multi-Key Gemini Router & VLM  │    │ 3. 5× CatBoost Regressors         │
        │ 4. Food Category Density & Volume │    │ 4. Atwater Factor Reconciliation  │
        └───────────────────────────────────┘    └───────────────────────────────────┘
```

### 📂 Clean Project Structure

```text
APP/
├── FrontEnd/                 # All Frontend UI assets (HTML, CSS, JS, images, icons)
├── BackEnd/                  # Python backend directly serving FrontEnd (Port 8000)
│   └── app.py                # Gateway API, static file server, proxy to DL & ML
├── Intelligence/             # AI & Intelligence Core
│   ├── DL/                   # Deep Learning / Vision endpoint (Port 8001)
│   │   ├── app.py            # FastAPI service for computer vision
│   │   ├── segmentation.py   # DeepLabV3+ semantic segmentation & plate metrology
│   │   ├── estimation.py     # Density category volume-to-gram modeling
│   │   ├── gemini_router.py  # Multi-key failover router & structured JSON parser
│   │   ├── vlm.py            # Multimodal vision-language model integration
│   │   ├── depth.py          # 3D plate depth map generation
│   │   └── models/           # DeepLabV3+ model weights & labels config
│   └── ML/                   # Machine Learning / Nutrition endpoint (Port 8002)
│       ├── app.py            # FastAPI service for nutritional regression
│       ├── predictor.py      # CatBoost inference & mass conservation engine
│       ├── preprocessor.py   # Text feature extraction & keywords
│       ├── schemas.py        # Pydantic request/response schemas
│       ├── *.cbm             # 5 trained CatBoost regression models
│       └── *.joblib          # Vectorizer, SVD, and keyword artifacts
├── tests/                    # Integration & unit test suites
├── start_services.ps1        # Unified launcher starting all 3 microservices
├── run_app.bat               # Windows batch launcher
├── supabase_schema.sql       # Database schema & migrations
└── requirements.txt          # Unified Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10 or 3.11** (64-bit)
- **Ollama** installed on your system with `qwen3-vl:2b`:
  ```powershell
  ollama pull qwen3-vl:2b
  ```

### 2. Environment Setup
Install the unified dependencies from the project root:
```powershell
py -3.11 -m venv .venv_run
.\.venv_run\Scripts\python.exe -m pip install --upgrade pip
.\.venv_run\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Launching Services
Run the automated launcher from PowerShell:
```powershell
.\start_services.ps1
```
This script validates Ollama, initializes both microservices, and opens the frontend UI:
* **Frontend UI & Nutrition Service:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Vision Service:** [http://127.0.0.1:8001](http://127.0.0.1:8001)
* **API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

---

## 🍛 IIIT Hyderabad Dataset Integration

NutraVision integrates empirical data from **CVIT IIIT Hyderabad (ICVGIP 2025 Indian Thali Dataset)** and **ICMR-NIN (National Institute of Nutrition)**:

1. **Physical Portion Bounds:** Over 3,278 physically weighed Indian servings constrain computer-vision volume estimation:
   * Flatbreads (*Phulka / Roti*): bounded to $30\text{--}80\text{ g}$ (preventing 200g loaf hallucinations).
   * *Papad*: bounded to $\le 25\text{ g}$.
   * *Chutneys & Pickles*: bounded to $\le 45\text{ g}$.
2. **Context-Aware Preparation (3-Tier Resolution):**
   * *Explicit:* `"grilled chicken"` $\to$ grilled.
   * *Contextual:* Rice inside a `"Biryani"` $\to$ boiled with ghee/spices; veggies in a `"Salad"` $\to$ raw.
   * *Smart Default:* Inherent food physical state.

---

## 🧪 Running the Test Suite

Validate all endpoints, models, CORS preflight, and dual-image inference:
```powershell
.\.venv_run\Scripts\python.exe tests/test_suite.py
```

Run Indian food macro accuracy benchmarks:
```powershell
.\.venv_run\Scripts\python.exe tests/test_indian_foods.py
```

---

## 🛡️ License
Academic and research use under MIT License. Models and datasets subject to respective upstream licenses (FoodSeg103, IIIT-H CVIT, ICMR-NIN).
