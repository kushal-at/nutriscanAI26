/**
 * NutriScanAI — Deployment Configuration
 * --------------------------------------
 * When running locally with start_services.ps1 (Ports 8000, 8001, 8002),
 * leave DL_URL and ML_URL empty ("") so requests flow through the local BackEnd gateway.
 *
 * When deploying FrontEnd to Vercel and ML/DL to cloud hosts:
 * Set DL_URL and ML_URL to your deployed service URLs.
 */
window.NUTRISCAN_CONFIG = {
    DL_URL: "",
    ML_URL: ""
};
