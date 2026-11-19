#!/bin/bash
set -e

# ==============================================================================
# NutriScanAI Production Cloud Container Startup
# Boots unified NutriScanAI full-stack service (ML Nutrition, 3D Vision, and UI)
# Fits cleanly inside 512MB RAM (Render, Railway, Hugging Face Spaces, Cloud Run)
# ==============================================================================

echo "=========================================================="
echo " Starting NutriScanAI Cloud Stack"
echo "=========================================================="

PUBLIC_PORT="${PORT:-7860}"
echo "Launching NutriScanAI Unified Service on 0.0.0.0:${PUBLIC_PORT}..."
exec python -m uvicorn app:app --host 0.0.0.0 --port "${PUBLIC_PORT}"
