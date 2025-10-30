<# 
  quickstart_windows.ps1
  ----------------------
  1) Create & activate venv
  2) pip install -r requirements.txt
  3) Install the Invoke-StopMotion2D cmdlet (using venv python)
  4) Run one pipeline pass on a sample clip

  Usage:
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    .\scripts\quickstart_windows.ps1 -Clip ".\clips\cropped\test1.mp4" -Out ".\out"
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$false)][string]$Clip = ".\clips\cropped\test1.mp4",
  [Parameter(Mandatory=$false)][string]$Out  = ".\out",
  [Parameter(Mandatory=$false)][double]$K        = 10,
  [Parameter(Mandatory=$false)][double]$SigmaT   = 0.015,
  [Parameter(Mandatory=$false)][double]$SigmaDeg = 0.4,
  [Parameter(Mandatory=$false)][int]$FitAuto     = 1,
  [Parameter(Mandatory=$false)][int]$RetimeVideo = 1,
  [Parameter(Mandatory=$false)][double]$Grain    = 0.01,
  [Parameter(Mandatory=$false)][double]$JitterPx = 0.2,
  [Parameter(Mandatory=$false)][double]$FpsOverride = 30
)

$ErrorActionPreference = "Stop"

Write-Host "== StopMotionFilter quickstart (Windows) ==" -ForegroundColor Cyan

# Ensure standard folders exist
$folders = @(".\clips", ".\clips\raw", ".\clips\cropped", ".\out", ".\logs", ".\scripts")
foreach ($f in $folders) { if (-not (Test-Path $f)) { New-Item -ItemType Directory -Path $f | Out-Null } }

# 1) Create & activate venv
if (-not (Test-Path ".\.venv")) {
  Write-Host "[1/4] Creating venv ..." -ForegroundColor Yellow
  python -m venv .venv
}
Write-Host "[2/4] Activating venv ..." -ForegroundColor Yellow
. .\.venv\Scripts\Activate.ps1

# 2) Install requirements
Write-Host "[3/4] Installing requirements ..." -ForegroundColor Yellow
pip install -r requirements.txt

# 3) Install the cmdlet using venv python
$Py = ".\.venv\Scripts\python.exe"
Write-Host "[4/4] Installing cmdlet Invoke-StopMotion2D ..." -ForegroundColor Yellow
.\Install-StopMotion2D.ps1 -ProjectDir (Get-Location).Path -PythonExe $Py

# Run once
$run = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out | Out-Null }
$outRun = Join-Path $Out "run_$run"

Write-Host ">> Running pipeline into $outRun" -ForegroundColor Green
Invoke-StopMotion2D `
  -InputVideo $Clip `
  -OutDir $outRun `
  -FitAuto $FitAuto `
  -K $K `
  -SigmaT $SigmaT `
  -SigmaDeg $SigmaDeg `
  -RetimeVideo $RetimeVideo `
  -Grain $Grain `
  -JitterPx $JitterPx `
  -OutFps $FpsOverride `
  -FpsOverride $FpsOverride

Write-Host "Done. Outputs in: $outRun" -ForegroundColor Green

