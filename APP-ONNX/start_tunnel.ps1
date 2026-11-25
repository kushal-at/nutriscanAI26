<#
.SYNOPSIS
    Starts a free Cloudflare Quick Tunnel for NutriScanAI.
    Exposes your local instance (http://localhost:8000) to a public HTTPS URL.
#>

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Launching Cloudflare Tunnel for NutriScanAI (Port 8000) " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Make sure services are running via: .\start_services.ps1`n" -ForegroundColor Yellow

cloudflared tunnel --url http://localhost:8000
