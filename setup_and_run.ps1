<#
.SYNOPSIS
    Script de automatización — Taller 1: Dynamic Programming (MilanTaxi)
.DESCRIPTION
    Verifica/crea el entorno virtual, instala dependencias, registra el kernel
    de Jupyter y lanza el notebook principal (milan_taxi_es.ipynb).
.EXAMPLE
    .\setup_and_run.ps1
#>

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR      = Join-Path $PROJECT_ROOT ".venv"
$NOTEBOOK_PATH = Join-Path $PROJECT_ROOT "notebooks" "milan_taxi_es.ipynb"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Taller 1 — Dynamic Programming (MilanTaxi)" -ForegroundColor Cyan
Write-Host "  Asistente de configuración y ejecución" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Detectar / crear el entorno virtual
# ---------------------------------------------------------------------------
Write-Host "`n[1/5] Verificando entorno virtual..." -ForegroundColor Green

if (-not (Test-Path (Join-Path $VENV_DIR "Scripts" "python.exe"))) {
    Write-Host "  [+] Creando .venv con Python 3.12..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] No se pudo crear el entorno virtual." -ForegroundColor Red
        exit 1
    }
    Write-Host "  [+] Entorno virtual creado en $VENV_DIR" -ForegroundColor Green
} else {
    Write-Host "  [+] Entorno virtual encontrado en $VENV_DIR" -ForegroundColor Green
}

$PYTHON = Join-Path $VENV_DIR "Scripts" "python.exe"
$PIP    = Join-Path $VENV_DIR "Scripts" "pip.exe"

# ---------------------------------------------------------------------------
# 2. Actualizar pip e instalar dependencias
# ---------------------------------------------------------------------------
Write-Host "`n[2/5] Instalando dependencias..." -ForegroundColor Green

& $PIP install --upgrade pip -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] No se pudo actualizar pip, continuando..." -ForegroundColor Yellow
}

& $PIP install -e "$PROJECT_ROOT" -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Fallo al instalar el paquete del proyecto." -ForegroundColor Red
    exit 1
}

# Dependencias adicionales para el notebook
& $PIP install matplotlib seaborn jupyter ipywidgets notebook -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Fallo al instalar dependencias del notebook." -ForegroundColor Red
    exit 1
}

Write-Host "  [+] Dependencias instaladas correctamente." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 3. Verificar que el módulo se importa correctamente
# ---------------------------------------------------------------------------
Write-Host "`n[3/5] Verificando importación del módulo..." -ForegroundColor Green

& $PYTHON -c "import sys; sys.path.insert(0, '$PROJECT_ROOT\\src'); from rl_project.envs.milan_taxi import MilanTaxiEnv; print('  [+] Módulo rl_project importado correctamente.')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] El módulo rl_project no se puede importar." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Registrar kernel de Jupyter para el .venv
# ---------------------------------------------------------------------------
Write-Host "`n[4/5] Registrando kernel de Jupyter..." -ForegroundColor Green

& $PYTHON -m ipykernel install --user --name ".venv" --display-name ".venv (Taller1)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [+] Kernel '.venv (Taller1)' registrado." -ForegroundColor Green
} else {
    Write-Host "  [WARN] No se pudo registrar el kernel (continuando de todas formas)." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 5. Lanzar Jupyter Notebook
# ---------------------------------------------------------------------------
Write-Host "`n[5/5] Lanzando Jupyter Notebook..." -ForegroundColor Green

if (-not (Test-Path $NOTEBOOK_PATH)) {
    Write-Host "  [WARN] No se encontró $NOTEBOOK_PATH" -ForegroundColor Yellow
}

Write-Host "  [+] Abriendo: $NOTEBOOK_PATH" -ForegroundColor Cyan
Write-Host "  [+] Notebook -> http://localhost:8888/notebooks/notebooks/milan_taxi_es.ipynb" -ForegroundColor Cyan
Write-Host "`n"  -ForegroundColor Cyan

& $PYTHON -m notebook "$NOTEBOOK_PATH"