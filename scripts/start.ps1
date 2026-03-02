param(
    [switch]$NoMilvus,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "No se encontro '$Name' en PATH."
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"

$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No existe backend/.venv. Ejecuta primero .\scripts\install.ps1"
}

if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    throw "No existe frontend/node_modules. Ejecuta primero .\scripts\install.ps1"
}

Require-Command "powershell"
Require-Command "npm"

$backendEnv = Join-Path $backendDir ".env"
$backendEnvExample = Join-Path $backendDir ".env.example"
if ((-not (Test-Path $backendEnv)) -and (Test-Path $backendEnvExample)) {
    Copy-Item $backendEnvExample $backendEnv
}

$frontendEnv = Join-Path $frontendDir ".env"
$frontendEnvExample = Join-Path $frontendDir ".env.example"
if ((-not (Test-Path $frontendEnv)) -and (Test-Path $frontendEnvExample)) {
    Copy-Item $frontendEnvExample $frontendEnv
}

if (-not $NoMilvus) {
    Require-Command "docker"
    Write-Host "==> Levantando Milvus (docker compose)" -ForegroundColor Cyan
    Push-Location $backendDir
    try {
        & docker compose -f "docker-compose.milvus.yml" up -d
    }
    finally {
        Pop-Location
    }
}

Write-Host "==> Iniciando backend en nueva consola" -ForegroundColor Cyan
$backendCmd = "& '$venvPython' -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload"
Start-Process -FilePath "powershell" -WorkingDirectory $backendDir -ArgumentList "-NoExit", "-Command", $backendCmd | Out-Null

Write-Host "==> Iniciando frontend en nueva consola" -ForegroundColor Cyan
$frontendCmd = "npm run dev -- --host 0.0.0.0 --port 5173"
Start-Process -FilePath "powershell" -WorkingDirectory $frontendDir -ArgumentList "-NoExit", "-Command", $frontendCmd | Out-Null

Write-Host ""
Write-Host "Servicios iniciados:" -ForegroundColor Green
Write-Host "- Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "- Backend:  http://localhost:8001" -ForegroundColor Green
Write-Host "- Health:   http://localhost:8001/health" -ForegroundColor Green

if (-not $NoBrowser) {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:5173"
}
