# 🥗 NutraVision / NutriScanAI

This repository is organized into application and reference directories:

## 📁 Repository Structure

### 🚀 [`APP/`](./APP) — Production Application
Contains the complete, self-contained NutraVision platform:
* **[`APP/FrontEnd/`](./APP/FrontEnd/)**: Responsive web application UI (dual-camera food scanner, nutrition dashboard, meal history, Supabase sync).
* **[`APP/ML/`](./APP/ML/)**: Nutrition regression microservice (Port 8000) powered by 5 trained CatBoost models (`energy_kcal`, `carb_g`, `protein_g`, `fat_g`, `fibre_g`), 3-tier preparation resolution, and Atwater factor reconciliation.
* **[`APP/vision/`](./APP/vision/)**: Vision microservice (Port 8001) with PyTorch DeepLabV3+ semantic segmentation, tilt-compensated plate rim metrology, multi-key Google Gemini router, and volume-to-gram estimation.
* **[`APP/tests/`](./APP/tests/)**: Automated test suite for accuracy guardrails, endpoints, and failover router.
* **[`APP/app.py`](./APP/app.py)**: Main application entry point.
* **[`APP/start_services.ps1`](./APP/start_services.ps1)** & **[`APP/run_app.bat`](./APP/run_app.bat)**: Automated service launchers.
* **[`APP/supabase_schema.sql`](./APP/supabase_schema.sql)**: Database migrations and schema for user profiles, meals, and diary logs.

To start the application:
```powershell
cd APP
.\start_services.ps1
```
Or simply double-click `run_app.bat`.

---

### 📚 [`REF/`](./REF) — Reference Materials & Research
Contains background research, pitch materials, design assets, and legacy experiments:
* **[`REF/BasePaper.pdf`](./REF/BasePaper.pdf)**: Base academic research paper.
* **[`REF/OtherDocs/`](./REF/OtherDocs/)**: Final year engineering project presentation slides, pitch deck, model papers, and LoRA weights.
* **[`REF/uiRedesign/`](./REF/uiRedesign/)**: UI design mockups, visual direction guides, and screenshot reference contact sheets.
* **[`REF/Temp/`](./REF/Temp/)**: Legacy dataset builders, training scripts, ablation studies, and evaluation experiments.
* **[`REF/scratch/`](./REF/scratch/)**: Scratch agent exploration scripts.
* **[`REF/logs/`](./REF/logs/)**: Archival server run logs.
* **[`REF/skills/`](./REF/skills/)**: Auxiliary agent skill definitions.
