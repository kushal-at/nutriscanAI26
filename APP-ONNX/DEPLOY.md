# 🚀 Deploying NutriScanAI (Vercel + Hugging Face Spaces — No Docker)

This guide deploys the 3 tiers completely free without Docker:
1. **ML (Nutrition Regression)** → **Hugging Face Spaces** (`Gradio` SDK)
2. **DL (Food Vision & Segmentation)** → **Hugging Face Spaces** (`Gradio` SDK)
3. **FrontEnd (Web App)** → **Vercel** (Static CDN)

---

## Step 1: Deploy ML to Hugging Face Spaces (No Docker)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Name it e.g. `nutriscan-ml`.
3. Select **SDK: `Gradio`** (Free CPU Basic, 16 GB RAM — **do not select Docker**).
4. Upload or push the contents of **`APP/Intelligence/ML/`**:
   - `app.py`
   - `predictor.py`, `preprocessor.py`, `schemas.py`
   - `requirements.txt`
   - `README.md` (already includes `sdk: gradio` frontmatter)
   - All `*.cbm` and `*.joblib` model files
5. Once the Space shows **Running**, copy its direct URL:
   `https://<your-hf-username>-nutriscan-ml.hf.space`
   *(Verify by visiting `<url>/health` — it should return `{"status": "ok", ...}`).*

---

## Step 2: Deploy DL (Vision) to Hugging Face Spaces (No Docker)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Name it e.g. `nutriscan-dl`.
3. Select **SDK: `Gradio`** (Free CPU Basic, 16 GB RAM — **do not select Docker**).
4. Upload or push the contents of **`APP/Intelligence/DL/`**:
   - `app.py`, `segmentation.py`, `estimation.py`, `vlm.py`, `gemini_router.py`, `depth.py`
   - `food_density.json`, `recipe_decomposer.json`
   - `models/FoodSeg103/best_model.pth` & `labels_config.json`
   - `requirements.txt` (uses `opencv-python-headless` for Linux)
   - `README.md` (already includes `sdk: gradio` frontmatter)
   - *(Note: `.gitignore` automatically excludes `gemini_keys.json` and `.gemini_vault.key` so secrets are never leaked).*
5. In your DL Space, go to **Settings → Variables and secrets → New secret**:
   - Name: `GEMINI_API_KEYS`
   - Value: `AIzaSy...` (one or comma-separated multiple Gemini API keys)
6. Once the Space shows **Running**, copy its direct URL:
   `https://<your-hf-username>-nutriscan-dl.hf.space`
   *(Verify by visiting `<url>/health`).*

---

## Step 3: Deploy FrontEnd to Vercel

1. Open [`APP/FrontEnd/config.js`](./FrontEnd/config.js) and paste your two Hugging Face Space URLs:
   ```javascript
   window.NUTRISCAN_CONFIG = {
       DL_URL: "https://<your-hf-username>-nutriscan-dl.hf.space",
       ML_URL: "https://<your-hf-username>-nutriscan-ml.hf.space"
   };
   ```
   *(Leave them as `""` when running locally with `start_services.ps1`).*
2. Deploy to **Vercel**:
   - **Via Vercel Dashboard (GitHub)**: Import your repository, set **Root Directory** to `APP/FrontEnd`, Framework Preset = `Other`, and click **Deploy**.
   - **Via Vercel CLI**:
     ```powershell
     cd APP\FrontEnd
     npx vercel --prod
     ```
3. Open your live Vercel URL — it connects directly to your DL and ML Hugging Face Spaces with no 10-second serverless timeout!
