<#
.SYNOPSIS
    NutraVision Multi-Service Launcher for Windows.
    Starts 3 dedicated microservices:
    - Deep Learning (DL / Vision API): Port 8001
    - Machine Learning (ML / Nutrition API): Port 8002
    - FrontEnd & BackEnd Gateway: Port 8000
#>

$ErrorActionPreference = "Stop"

$MainProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DLProjectRoot = Join-Path $MainProjectRoot "Intelligence\DL"
$MLProjectRoot = Join-Path $MainProjectRoot "Intelligence\ML"
$BackEndProjectRoot = Join-Path $MainProjectRoot "BackEnd"
$FrontEndProjectRoot = Join-Path $MainProjectRoot "FrontEnd"

if (-not (Test-Path -LiteralPath $DLProjectRoot)) {
    Write-Error "DL project directory not found at $DLProjectRoot."
    exit 1
}
if (-not (Test-Path -LiteralPath $MLProjectRoot)) {
    Write-Error "ML project directory not found at $MLProjectRoot."
    exit 1
}
if (-not (Test-Path -LiteralPath $BackEndProjectRoot)) {
    Write-Error "BackEnd project directory not found at $BackEndProjectRoot."
    exit 1
}

# Locate Python environment
$PythonCandidates = @(
    (Join-Path $MainProjectRoot ".venv_run\Scripts\python.exe"),
    (Join-Path $MainProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $MainProjectRoot) ".venv_run\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $MainProjectRoot) ".venv\Scripts\python.exe")
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
    Write-Error "Python executable not found. Please install Python or set up a virtual environment."
    exit 1
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "       NUTRAVISION AI - FOOD & NUTRITION PLATFORM                " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Using Python: $Python" -ForegroundColor DarkGray

# 1. Check DL Model Artifacts
$DLWeights = Join-Path $DLProjectRoot "models\FoodSeg103\best_model.pth"
if (-not (Test-Path -LiteralPath $DLWeights)) {
    Write-Error "Vision model weights missing at $DLWeights."
    exit 1
}

# 2. Check ML CatBoost Model Artifacts
$CatBoostModel = Join-Path $MLProjectRoot "energy_kcal.cbm"
if (-not (Test-Path -LiteralPath $CatBoostModel)) {
    Write-Error "Nutrition CatBoost models missing in $MLProjectRoot."
    exit 1
}

# 3. Check Gemini API Key Pool Status
Write-Host "`n[1/4] Checking Gemini AI Key Router..." -ForegroundColor Yellow
$GeminiKeysFile = Join-Path $DLProjectRoot "gemini_keys.json"
$GeminiKeyCount = 0
if (Test-Path -LiteralPath $GeminiKeysFile) {
    try {
        $keysData = Get-Content -LiteralPath $GeminiKeysFile -Raw | ConvertFrom-Json
        if ($keysData.keys) {
            $GeminiKeyCount = $keysData.keys.Count
        }
    } catch {}
}
if ($env:GEMINI_API_KEYS) {
    $envKeys = $env:GEMINI_API_KEYS -split "[,;\r\n]+" | Where-Object { $_.Trim() -ne "" }
    $GeminiKeyCount = [Math]::Max($GeminiKeyCount, $envKeys.Count)
} elseif ($env:GEMINI_API_KEY) {
    $GeminiKeyCount = [Math]::Max($GeminiKeyCount, 1)
}

if ($GeminiKeyCount -gt 0) {
    Write-Host "  -> Gemini Key Router ready ($GeminiKeyCount key(s) registered with round-robin rotation & 429 failover)." -ForegroundColor Green
} else {
    Write-Host "  -> No Gemini API keys found. Vision service will use DeepLabV3+ geometric segmentation fallback." -ForegroundColor DarkGray
}

# 4. Check Local Ollama Service for NutriBot AI
Write-Host "`n[2/4] Checking Ollama Agent Service (qwen3-vl:2b)..." -ForegroundColor Yellow
$OllamaCandidate = "C:\Users\sazku\AppData\Local\Programs\Ollama\ollama.exe"
if (-not (Test-Path -LiteralPath $OllamaCandidate)) {
    $cmd = Get-Command "ollama" -ErrorAction SilentlyContinue
    if ($cmd) { $OllamaCandidate = $cmd.Source }
}

$OllamaListening = $false
try {
    $tcp = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
    if ($tcp) { $OllamaListening = $true }
} catch {}

if ($OllamaListening) {
    Write-Host "  -> Ollama service is active on port 11434 (Ready for NutriBot)." -ForegroundColor Green
} elseif (Test-Path -LiteralPath $OllamaCandidate) {
    Write-Host "  -> Launching Ollama in background ($OllamaCandidate serve)..." -ForegroundColor Cyan
    Start-Process -FilePath $OllamaCandidate -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 2
    Write-Host "  -> Ollama launched successfully for NutriBot." -ForegroundColor Green
} else {
    Write-Host "  -> Ollama executable not found. Running NutriBot with intelligent clinical nutrition fallback." -ForegroundColor DarkGray
}

# 5. Check & safely release ports 8000, 8001, and 8002
Write-Host "`n[3/4] Checking and clearing ports 8000, 8001, and 8002..." -ForegroundColor Yellow
$env:PYTHONUNBUFFERED = "1"
@(8000, 8001, 8002) | ForEach-Object {
    $port = $_
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($connections) {
        $pids = $connections.OwningProcess | Select-Object -Unique
        foreach ($p in $pids) {
            if ($p -gt 4) {
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

# 6. Prepare Logs Directory & Launch All 3 Services
Write-Host "`n[4/4] Launching 3 Dedicated Microservices..." -ForegroundColor Yellow
$LogsDir = Join-Path $MainProjectRoot "logs"
if (-not (Test-Path -LiteralPath $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

$DLOutLog = Join-Path $LogsDir "dl_stdout.log"
$DLErrLog = Join-Path $LogsDir "dl_stderr.log"
$MLOutLog = Join-Path $LogsDir "ml_stdout.log"
$MLErrLog = Join-Path $LogsDir "ml_stderr.log"
$BackEndOutLog = Join-Path $LogsDir "backend_stdout.log"
$BackEndErrLog = Join-Path $LogsDir "backend_stderr.log"

# Port 8001: Deep Learning (DL / Vision API)
Write-Host "  -> [Port 8001] Starting Deep Learning Service (Intelligence/DL)..." -ForegroundColor Cyan
$DLProcess = Start-Process -FilePath $Python -ArgumentList "-u -m uvicorn app:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 65" -WorkingDirectory $DLProjectRoot -RedirectStandardOutput $DLOutLog -RedirectStandardError $DLErrLog -PassThru

# Port 8002: Machine Learning (ML / Nutrition API)
Write-Host "  -> [Port 8002] Starting Machine Learning Service (Intelligence/ML)..." -ForegroundColor Cyan
$MLProcess = Start-Process -FilePath $Python -ArgumentList "-u -m uvicorn app:app --host 0.0.0.0 --port 8002 --timeout-keep-alive 65" -WorkingDirectory $MLProjectRoot -RedirectStandardOutput $MLOutLog -RedirectStandardError $MLErrLog -PassThru

# Port 8000: FrontEnd & BackEnd Gateway
Write-Host "  -> [Port 8000] Starting FrontEnd & BackEnd Gateway (BackEnd)..." -ForegroundColor Cyan
$BackEndProcess = Start-Process -FilePath $Python -ArgumentList "-u -m uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 65" -WorkingDirectory $BackEndProjectRoot -RedirectStandardOutput $BackEndOutLog -RedirectStandardError $BackEndErrLog -PassThru

Start-Sleep -Seconds 3

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "            ALL 3 SERVICES ACTIVE AND READY!                    " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  * [Port 8000] FrontEnd UI & Gateway:  http://127.0.0.1:8000/" -ForegroundColor White
Write-Host "  * [Port 8001] Deep Learning API (DL): http://127.0.0.1:8001/docs" -ForegroundColor White
Write-Host "  * [Port 8002] Machine Learning (ML):  http://127.0.0.1:8002/docs" -ForegroundColor White
Write-Host "  * NutriBot Assistant:                 http://127.0.0.1:11434 (qwen3-vl:2b)" -ForegroundColor White
Write-Host "  * Output Logs:                        $LogsDir" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate all 3 services." -ForegroundColor DarkGray

# Open default browser to Frontend UI
try {
    Start-Process "http://127.0.0.1:8000/" -ErrorAction SilentlyContinue
} catch {}

try {
    while ($true) {
        if ($DLProcess.HasExited) {
            Write-Host "Deep Learning Service (8001) exited unexpectedly (Code: $($DLProcess.ExitCode)). Check $DLErrLog" -ForegroundColor Red
            break
        }
        if ($MLProcess.HasExited) {
            Write-Host "Machine Learning Service (8002) exited unexpectedly (Code: $($MLProcess.ExitCode)). Check $MLErrLog" -ForegroundColor Red
            break
        }
        if ($BackEndProcess.HasExited) {
            Write-Host "BackEnd Gateway (8000) exited unexpectedly (Code: $($BackEndProcess.ExitCode)). Check $BackEndErrLog" -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host "`nStopping all background services..." -ForegroundColor Yellow
    if ($DLProcess -and -not $DLProcess.HasExited) { Stop-Process -Id $DLProcess.Id -Force -ErrorAction SilentlyContinue }
    if ($MLProcess -and -not $MLProcess.HasExited) { Stop-Process -Id $MLProcess.Id -Force -ErrorAction SilentlyContinue }
    if ($BackEndProcess -and -not $BackEndProcess.HasExited) { Stop-Process -Id $BackEndProcess.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "All 3 services stopped cleanly." -ForegroundColor Green
}
