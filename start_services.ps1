<#
.SYNOPSIS
    NutraVision Unified Multi-Service Launcher for Windows 11.
    Starts the Vision API (port 8001) and Nutrition ML API / Frontend (port 8000).
#>

$ErrorActionPreference = "Stop"

$MainProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VisionProjectRoot = Join-Path $MainProjectRoot "vision"
if (-not (Test-Path -LiteralPath $VisionProjectRoot)) {
    Write-Error "Vision project directory not found at $VisionProjectRoot. Please ensure the vision/ directory exists."
    exit 1
}

# Locate Python environment flexibly: .venv_run, .venv, or system python / py launcher
$PythonCandidates = @(
    (Join-Path $MainProjectRoot ".venv_run\Scripts\python.exe"),
    (Join-Path $MainProjectRoot ".venv\Scripts\python.exe")
)
$Python = $PythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $Python) {
    $pyCmd = Get-Command "python" -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $Python = $pyCmd.Source
    } else {
        $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            $Python = $pyLauncher.Source
        }
    }
}

if (-not $Python) {
    Write-Error "Python executable not found in .venv_run, .venv, or system PATH. Please set up Python or virtual environment."
    exit 1
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "       NUTRAVISION AI - FOOD & NUTRITION PLATFORM                " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Using Python: $Python" -ForegroundColor DarkGray

# 1. Check Vision Model Artifacts
$VisionWeights = Join-Path $VisionProjectRoot "models\FoodSeg103\best_model.pth"
if (-not (Test-Path -LiteralPath $VisionWeights)) {
    Write-Error "Vision model weights missing at $VisionWeights. Check vision/models/FoodSeg103."
    exit 1
}

# 2. Check CatBoost Model Artifacts
$MLDir = Join-Path $MainProjectRoot "ML"
$CatBoostModel = Join-Path $MLDir "energy_kcal.cbm"
if (-not (Test-Path -LiteralPath $CatBoostModel)) {
    Write-Error "Nutrition CatBoost models missing in $MLDir. Run python train/train_catboost.py first."
    exit 1
}

# 3. Check Ollama Connectivity (Port 11434)
Write-Host "`n[1/3] Checking Ollama Service..." -ForegroundColor Yellow
$OllamaRunning = $false
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($ollamaCheck -and $ollamaCheck.models) {
        $modelNames = $ollamaCheck.models | ForEach-Object { $_.name }
        if ($modelNames -contains "qwen3-vl:2b") {
            Write-Host "  -> Ollama is online with qwen3-vl:2b ready." -ForegroundColor Green
            $OllamaRunning = $true
        } else {
            Write-Host "  -> Ollama is running, but 'qwen3-vl:2b' was not found in: $($modelNames -join ', ')." -ForegroundColor Yellow
            Write-Host "     Run: 'ollama pull qwen3-vl:2b' for VLM portion estimations." -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "  -> Ollama not detected at http://127.0.0.1:11434." -ForegroundColor Yellow
    Write-Host "     Vision service will use DeepLabV3+ geometric segmentation fallback." -ForegroundColor DarkGray
}

# 4. Check & safely release ports 8000 and 8001
Write-Host "`n[2/3] Checking ports 8000 and 8001..." -ForegroundColor Yellow
$env:PYTHONUNBUFFERED = "1"
@(8000, 8001) | ForEach-Object {
    $port = $_
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($connections) {
        $pids = $connections.OwningProcess | Select-Object -Unique
        foreach ($p in $pids) {
            if ($p -gt 4) { # Skip System and Idle processes
                $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                if ($proc) {
                    Write-Host "  -> Releasing port $port (stopping process '$($proc.ProcessName)' PID $p)..." -ForegroundColor DarkYellow
                    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                }
            }
        }
    }
}
Start-Sleep -Seconds 1

# 5. Prepare Logs Directory & Launch Services
Write-Host "`n[3/3] Launching Services..." -ForegroundColor Yellow
$LogsDir = Join-Path $MainProjectRoot "logs"
if (-not (Test-Path -LiteralPath $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}
$VisionOutLog = Join-Path $LogsDir "vision_stdout.log"
$VisionErrLog = Join-Path $LogsDir "vision_stderr.log"
$NutritionOutLog = Join-Path $LogsDir "nutrition_stdout.log"
$NutritionErrLog = Join-Path $LogsDir "nutrition_stderr.log"
Write-Host "  -> Redirecting unbuffered output logs to: $LogsDir" -ForegroundColor DarkGray

# Start Vision API on Port 8001 (unbuffered)
Write-Host "  -> Starting Vision Service on port 8001 (vision)..." -ForegroundColor Cyan
$VisionProcess = Start-Process -FilePath $Python -ArgumentList "-u -m uvicorn app:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 65" -WorkingDirectory $VisionProjectRoot -RedirectStandardOutput $VisionOutLog -RedirectStandardError $VisionErrLog -PassThru

# Start Nutrition & Frontend API on Port 8000 (unbuffered)
Write-Host "  -> Starting Nutrition Service & Frontend on port 8000 (ML)..." -ForegroundColor Cyan
$NutritionProcess = Start-Process -FilePath $Python -ArgumentList "-u -m uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 65" -WorkingDirectory $MainProjectRoot -RedirectStandardOutput $NutritionOutLog -RedirectStandardError $NutritionErrLog -PassThru

Start-Sleep -Seconds 3

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "            SERVICES ARE ACTIVE AND READY TO USE!               " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  * Frontend UI:             http://127.0.0.1:8000/" -ForegroundColor White
Write-Host "  * Nutrition API Docs:      http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  * Vision API Docs:         http://127.0.0.1:8001/docs" -ForegroundColor White
Write-Host "  * Logs Directory:          $LogsDir" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate both services." -ForegroundColor DarkGray

# Open default browser to Frontend UI
try {
    Start-Process "http://127.0.0.1:8000/" -ErrorAction SilentlyContinue
} catch {}

try {
    while ($true) {
        if ($VisionProcess.HasExited) {
            Write-Host "Vision Service exited unexpectedly (Code: $($VisionProcess.ExitCode)). Check $VisionErrLog" -ForegroundColor Red
            break
        }
        if ($NutritionProcess.HasExited) {
            Write-Host "Nutrition Service exited unexpectedly (Code: $($NutritionProcess.ExitCode)). Check $NutritionErrLog" -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host "`nStopping background services..." -ForegroundColor Yellow
    if ($VisionProcess -and -not $VisionProcess.HasExited) { Stop-Process -Id $VisionProcess.Id -Force -ErrorAction SilentlyContinue }
    if ($NutritionProcess -and -not $NutritionProcess.HasExited) { Stop-Process -Id $NutritionProcess.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Services stopped cleanly." -ForegroundColor Green
}
