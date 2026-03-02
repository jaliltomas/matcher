param(
    [switch]$InstallTorchCuda
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "No se encontro '$Name' en PATH. Instalalo y volve a ejecutar este script."
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"

if (-not (Test-Path $backendDir)) { throw "No existe backend/ en: $repoRoot" }
if (-not (Test-Path $frontendDir)) { throw "No existe frontend/ en: $repoRoot" }

Require-Command "python"
Require-Command "node"
Require-Command "npm"

Write-Host "==> Backend: creando/actualizando venv" -ForegroundColor Cyan
$venvDir = Join-Path $backendDir ".venv"
if (-not (Test-Path $venvDir)) {
    & python -m venv $venvDir
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No se encontro python del venv en: $venvPython"
}

& $venvPython -m pip install --upgrade pip setuptools wheel

if ($InstallTorchCuda) {
    Write-Host "==> Instalando PyTorch CUDA (cu121)" -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu121"
}

Write-Host "==> Instalando dependencias backend" -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $backendDir "requirements.txt")

$backendEnv = Join-Path $backendDir ".env"
$backendEnvExample = Join-Path $backendDir ".env.example"
if ((-not (Test-Path $backendEnv)) -and (Test-Path $backendEnvExample)) {
    Copy-Item $backendEnvExample $backendEnv
    Write-Host "==> Creado backend/.env desde .env.example" -ForegroundColor DarkCyan
}

Write-Host "==> Frontend: instalando dependencias npm" -ForegroundColor Cyan
Push-Location $frontendDir
try {
    & npm install
}
finally {
    Pop-Location
}

$frontendEnv = Join-Path $frontendDir ".env"
$frontendEnvExample = Join-Path $frontendDir ".env.example"
if ((-not (Test-Path $frontendEnv)) -and (Test-Path $frontendEnvExample)) {
    Copy-Item $frontendEnvExample $frontendEnv
    Write-Host "==> Creado frontend/.env desde .env.example" -ForegroundColor DarkCyan
}

Write-Host "" 
Write-Host "Instalacion lista." -ForegroundColor Green
Write-Host "Siguiente paso: .\scripts\start.ps1" -ForegroundColor Green
