<#
.SYNOPSIS
    Activa el entorno virtual, instala dependencias y lanza el notebook del Taller 1.

.DESCRIPTION
    Script de un solo paso para:
      1. Crear/activar .venv (Python 3.10+)
      2. Instalar el proyecto en modo editable (pip install -e .)
      3. Instalar dependencias de notebook (matplotlib, seaborn, jupyter, ipykernel, ipywidgets)
      4. Registrar el kernel Jupyter ".venv (Taller1)"
      5. Lanzar 'notebooks/milan_taxi_es.ipynb'

    Ideal para ejecutar al abrir la carpeta en VS Code:
        Terminal > New Terminal > .\activate.ps1
    O directamente desde PowerShell.

.EXAMPLE
    .\activate.ps1
#>

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR      = Join-Path $PROJECT_ROOT ".venv"
$NOTEBOOK_PATH = Join-Path $PROJECT_ROOT "notebooks" "milan_taxi_es.ipynb"
$PYTHON_EXE    = Join-Path $VENV_DIR "Scripts" "python.exe"
$PIP_EXE       = Join-Path $VENV_DIR "Scripts" "pip.exe"

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Taller 1 — Programación Dinámica (MilanTaxi)" -ForegroundColor Cyan
Write-Host "  Activación y lanzamiento automático" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Crear .venv si no existe
# ---------------------------------------------------------------------------
Write-Host "`n[1/5] Entorno virtual..." -ForegroundColor Green

if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "  [+] Creando .venv ..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] No se pudo crear .venv. Asegúrate de tener Python 3.10+." -ForegroundColor Red
        exit 1
    }
    Write-Host "  [+] .venv creado en $VENV_DIR" -ForegroundColor Green
} else {
    Write-Host "  [+] .venv ya existe en $VENV_DIR" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 2. Instalar dependencias
# ---------------------------------------------------------------------------
Write-Host "`n[2/5] Instalando dependencias..." -ForegroundColor Green

Write-Host "  [+] Actualizando pip e instalando setuptools..." -ForegroundColor Gray
& $PIP_EXE install --upgrade pip -q
& $PIP_EXE install setuptools -q

Write-Host "  [+] Instalando paquete del proyecto (pip install -e .) ..." -ForegroundColor Gray
& $PIP_EXE install -e "$PROJECT_ROOT" -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Falló pip install -e ." -ForegroundColor Red
    exit 1
}

Write-Host "  [+] Instalando dependencias para notebook..." -ForegroundColor Gray
& $PIP_EXE install matplotlib seaborn jupyter ipykernel ipywidgets notebook -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Falló instalación de dependencias." -ForegroundColor Red
    exit 1
}

Write-Host "  [+] Todas las dependencias instaladas." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 3. Verificar importación del módulo
# ---------------------------------------------------------------------------
Write-Host "`n[3/5] Verificando módulo rl_project..." -ForegroundColor Green

& $PYTHON_EXE -c "
import sys, os
sys.path.insert(0, os.path.abspath('$PROJECT_ROOT\\src'))
from rl_project.envs.milan_taxi import MilanTaxiEnv
from rl_project.models.mdp import build_model
from rl_project.agents.dynamic_programming import policy_iteration, value_iteration
from rl_project.evaluation import evaluate_policy
print('  [+] Todos los módulos se importan correctamente.')
"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Falló la verificación de importación del módulo." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Registrar kernel de Jupyter
# ---------------------------------------------------------------------------
Write-Host "`n[4/5] Registrando kernel Jupyter..." -ForegroundColor Green

& $PYTHON_EXE -m ipykernel install --user --name ".venv" --display-name ".venv (Taller1)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [+] Kernel '.venv (Taller1)' registrado." -ForegroundColor Green
} else {
    Write-Host "  [WARN] No se pudo registrar el kernel (continuando...)." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 5. Lanzar Jupyter Notebook
# ---------------------------------------------------------------------------
Write-Host "`n[5/5] Lanzando Jupyter Notebook..." -ForegroundColor Green

if (-not (Test-Path $NOTEBOOK_PATH)) {
    Write-Host "  [ERROR] No se encontró el notebook en:" -ForegroundColor Red
    Write-Host "         $NOTEBOOK_PATH" -ForegroundColor Red
    Write-Host "  Asegúrate de que el archivo existe en 'notebooks/milan_taxi_es.ipynb'." -ForegroundColor Red
    exit 1
}

Write-Host "  [+] Abriendo: $NOTEBOOK_PATH" -ForegroundColor Cyan
Write-Host "  [+] Una vez abierto, selecciona el kernel: '.venv (Taller1)'" -ForegroundColor Cyan
Write-Host "  [+] Luego ve a Cell > Run All para ejecutar todas las celdas." -ForegroundColor Cyan
Write-Host ""

# Lanzar notebook
& $PYTHON_EXE -m notebook "$NOTEBOOK_PATH"