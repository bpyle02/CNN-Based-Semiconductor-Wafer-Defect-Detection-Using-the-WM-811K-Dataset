param(
    [string]$Checkpoint = "checkpoints/defense_demo_cnn.pth"
)

$pytestCommand = Get-Command pytest -ErrorAction Stop
$pytestDir = Split-Path $pytestCommand.Source -Parent
$envRoot = Split-Path $pytestDir -Parent
$pythonExe = Join-Path $envRoot "python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Could not locate python.exe next to pytest at $pythonExe"
}

$scriptPath = Join-Path $PSScriptRoot "defense_smoke_demo.py"
& $pythonExe $scriptPath --checkpoint $Checkpoint
exit $LASTEXITCODE
