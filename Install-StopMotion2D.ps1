<#
.SYNOPSIS
  Installs a PowerShell cmdlet `Invoke-StopMotion2D` for the 2D stop-motion pipeline.

.DESCRIPTION
  Adds a function to your current PowerShell profile (and session) that wraps the Python orchestrator.
  Make sure Python is on PATH and you installed dependencies:
    pip install opencv-python mediapipe numpy pandas
#>

param(
  [string]$ProjectDir = (Get-Location).Path,
  [string]$PythonExe = "python"
)

$cmd = @"
function Invoke-StopMotion2D {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory=`$true)][string]`$InputVideo,
    [Parameter(Mandatory=`$true)][string]`$OutDir,
    [int]`$FitAuto = 1,
    [int]`$K = 8,
    [double]`$SigmaT = 0.02,
    [double]`$SigmaDeg = 0.5,
    [int]`$SmoothWin = 5,
    [int]`$RetimeVideo = 1,
    [double]`$Grain = 0.02,
    [double]`$JitterPx = 0.5,
    [double]`$OutFps = 0,
    [string]`$ClipId = "clip",
    [double]`$FpsOverride = 0
  )
  Push-Location "$ProjectDir"
  try {
    `$argsList = @(
      "pipeline_orchestrator.py",
      "--in", "$InputVideo",
      "--out_dir", "$OutDir",
      "--fit_auto", "$FitAuto",
      "--K", "$K",
      "--sigma_t", "$SigmaT",
      "--sigma_deg", "$SigmaDeg",
      "--smooth_win", "$SmoothWin",
      "--retime_video", "$RetimeVideo",
      "--grain", "$Grain",
      "--jitter_px", "$JitterPx",
      "--clip_id", "$ClipId",
      "--python", "$PythonExe"
    )
    if (`$OutFps -gt 0) { `$argsList += @("--out_fps", "$OutFps") }
    if (`$FpsOverride -gt 0) { `$argsList += @("--fps_override", "$FpsOverride") }
    & "$PythonExe" `$argsList
  } finally {
    Pop-Location
  }
}
"@

# Add to current session
Invoke-Expression $cmd
Write-Host "Cmdlet Invoke-StopMotion2D is available for this session." -ForegroundColor Green

# Persist to profile
$profileDir = Split-Path -Parent $PROFILE
if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Path $profileDir | Out-Null }
Add-Content -Path $PROFILE -Value $cmd
Write-Host "Persisted to your PowerShell profile: $PROFILE" -ForegroundColor Green

