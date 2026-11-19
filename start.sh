#!/bin/bash
set -e

# ==============================================================================
# NutriScanAI Cloud Container Startup Script
# Boots both the Vision AI service (8001) and ML Nutrition Regression & UI (PORT)
# ==============================================================================

echo "=========================================================="
echo " Starting NutriScanAI Cloud Stack"
echo "=========================================================="

# 1. Start Vision Service in background on internal port 8001
echo "[1/2] Launching Vision Service (FoodSeg103 + Gemini Router) on 127.0.0.1:8001..."
python -m uvicorn app:app --app-dir vision --host 127.0.0.1 --port 8001 &
VISION_PID=$!

# Wait briefly for Vision service socket
sleep 2

# 2. Start ML Nutrition & Frontend Service on public port (Hugging Face default: 7860, Render/Railway: $PORT)
PUBLIC_PORT="${PORT:-7860}"
echo "[2/2] Launching ML Nutrition & UI Service on 0.0.0.0:${PUBLIC_PORT}..."
python -m uvicorn app:app --host 0.0.0.0 --port "${PUBLIC_PORT}"

# Cleanup on exit
kill $VISION_PID 2>/dev/null || true
