# Taller 1 — Programación Dinámica en MilanTaxi

**Curso:** Aprendizaje por Refuerzo  
**Fecha:** Septiembre 2026  
**Raíz del proyecto:** `D:\ReforcingLearning\Taller1-Dynamic-Programming`

---

## 📋 Requisitos previos

- **Python ≥ 3.10** (probado con Python 3.12.13)
- **PowerShell 7+** (Windows) o **bash** (Linux/macOS)
- Conexión a internet para instalar dependencias

---

## 🚀 Inicio rápido (automatizado)

El script `setup_and_run.ps1` detecta/crea el entorno virtual, instala dependencias,
registra el kernel de Jupyter y abre el notebook principal.

### Windows (PowerShell)

```powershell
cd D:\ReforcingLearning\Taller1-Dynamic-Programming
.\setup_and_run.ps1
```

### Linux / macOS

```bash
cd /ruta/a/Taller1-Dynamic-Programming
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install matplotlib seaborn jupyter ipywidgets notebook
python -m ipykernel install --user --name ".venv"
jupyter notebook notebooks/milan_taxi_es.ipynb
```

---

## 🛠️ Instalación manual paso a paso

### 1. Crear y activar el entorno virtual

```powershell
# Windows (PowerShell)
cd D:\ReforcingLearning\Taller1-Dynamic-Programming
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar el paquete del proyecto y dependencias

```powershell
pip install --upgrade pip
pip install -e .
pip install matplotlib seaborn jupyter ipywidgets notebook
```

### 3. Registrar el kernel de Jupyter para el .venv

```powershell
python -m ipykernel install --user --name ".venv" --display-name ".venv (Taller1)"
```

### 4. Ejecutar el notebook

```powershell
python -m notebook notebooks/milan_taxi_es.ipynb
```

Una vez abierto, seleccionar el kernel **`.venv (Taller1)`** y ejecutar
**Cell → Run All**.
---

## ✅ Verificación de que todo funciona

Ejecute los tests unitarios rápidos (entorno y modelo MDP):

```powershell
cd D:\ReforcingLearning\Taller1-Dynamic-Programming
$env:PYTHONPATH="src"
.venv\Scripts\pytest tests/test_environment.py tests/test_mdp.py -q
```

Esperado: **24 passed** en ~0.5 s.

Para una verificación completa (incluyendo Policy Iteration y Value Iteration,
~60 s combinados):

```powershell
.venv\Scripts\pytest tests/ -q
```

Esperado: **42/42 PASS**.

---

## 📓 Notebooks

| Archivo | Idioma | Descripción |
|---|---|---|
| `notebooks/milan_taxi_es.ipynb` | 🇪🇸 Español | Versión principal del taller (implementación y análisis de PI y VI) |
| `notebooks/milan_taxi.ipynb` | 🇬🇧 Inglés | Versión original en inglés |

---

## 🎯 Objetivo

Implementar y analizar **Policy Iteration (PI)** y **Value Iteration (VI)** en el
entorno **MilanTaxi**. El entorno es una cuadrícula 5×5 con paredes internas
(4 puntos de interés), 500 estados y 6 acciones. Se estudian dos variantes:

- **Original:** Transiciones deterministas.
- **Estocástica:** Las acciones de movimiento tienen probabilidad de deslizamiento
  (0.1 para cada lado, izquierda/derecha).

---

## 📐 Formulación del MDP

| Componente | Descripción |
|---|---|
| **Espacio de estados S** | (fila, col, pasajero_idx, destino_idx) → 500 estados |
| **Espacio de acciones A** | SUR(0), NORTE(1), ESTE(2), OESTE(3), RECOGER(4), DEJAR(5) |
| **Transición P(s'\|s,a)** | Determinista (original) o deslizamiento estocástico |
| **Recompensa R(s,a)** | −1 por paso, −10 RECOGER/DEJAR fallido, +20 DEJAR exitoso |
| **Descuento γ** | 0.99 (salvo que se especifique otro) |

---

## 🧪 Resumen de resultados

| Entorno | Algoritmo | Iter/Barridos | V(inicio) | Tiempo (s) |
|---|---|---|---|---|
| Original | Policy Iteration | 11 | 1818.39 | ~5.0 |
| Original | Value Iteration | 2591 | 1818.39 | ~6.1 |
| Estocástico | Policy Iteration | 7 | 1771.30 | ~2.9 |
| Estocástico | Value Iteration | 2592 | 1771.30 | ~5.3 |

PI y VI producen la misma función de valor óptima (max diff < 1e-8).
La variante estocástica reduce V* en ~2.6 %.

## Evaluación empírica (1000 episodios)

| Política | Retorno medio | IC 95 % | Tasa de éxito |
|---|---|---|---|
| Original + PI | ~28.1 | ±0.6 | 100 % |
| Original + VI | ~28.1 | ±0.6 | 100 % |
| Estocástica + PI | ~5.5 | ±0.7 | 100 % |
| Estocástica + VI | ~5.5 | ±0.7 | 100 % |

---

## 🔑 Hallazgos clave

1. **PI es eficiente en iteraciones** (7-11) mientras que **VI es eficiente por
   barrido** (~2600 barridos baratos).
2. **La estocasticidad reduce el valor** — V(inicio) cae 2.6 %; el retorno
   empírico cae ~80 % (amplificado por el horizonte de 100 pasos).
3. **Ambos algoritmos convergen al mismo V*** (max diff < 1e-8).
4. **γ bajo** → convergencia rápida pero políticas miopes (V(inicio) negativo
   con γ=0.5).
5. **γ alto** (0.99) → V(inicio) alto pero ~2600 barridos en VI.

---

## 📁 Estructura del proyecto

```
Taller1-Dynamic-Programming/
├── .venv/                           # Entorno virtual (Python 3.12)
├── configs/                         # Configuraciones de experimentos (YAML)
├── notebooks/
│   ├── milan_taxi.ipynb             # Notebook original (inglés)
│   └── milan_taxi_es.ipynb          # Notebook del taller (español)
├── runs/                            # Resultados de experimentos (JSON, CSV)
├── src/
│   └── rl_project/
│       ├── agents/
│       │   └── dynamic_programming.py   # Implementación de PI y VI
│       ├── envs/
│       │   └── milan_taxi.py            # Entorno MilanTaxi
│       ├── models/
│       │   └── mdp.py                   # Constructor del MDP (P, R)
│       ├── evaluation.py                # Evaluación de políticas
│       ├── experiment.py                # Orquestación de experimentos
│       ├── policies.py                  # Utilidades de políticas
│       └── visualization.py             # Visualización interactiva
├── tests/
│   ├── test_environment.py          # 14 tests
│   ├── test_mdp.py                  # 10 tests
│   ├── test_policy_iteration.py     # 10 tests
│   └── test_value_iteration.py      # 8 tests
├── setup_and_run.ps1                # Script de automatización (PowerShell)
├── README.md                        # Este archivo
├── TALLER1_FINAL_REPORT.md
├── requirements.txt
└── pyproject.toml
```

---

## ⚠️ Limitaciones

- Horizonte truncado (100 pasos) → brecha entre valores teóricos y empíricos.
- Solo las acciones de movimiento son estocásticas; RECOGER/DEJAR deterministas.
- La PD tabular no escala más allá de este tamaño de problema.
- Los empates en Q (varias acciones con el mismo valor) causan ~6 % de desacuerdo.

---

## ♻️ Reproducibilidad

- Todas las semillas aleatorias están fijadas (seed=42 para evaluación,
  seed=0 para configuraciones).
- Resultados guardados en `runs/` como JSON + CSV con marcas de tiempo.
- Entorno virtual independiente `.venv` (Python 3.12.13).
- Código fuente en `src/rl_project/`, tests en `tests/`.

---

## 📝 Nota

El proyecto de referencia `D:\ReforcingLearning\rl-basics` **no fue modificado**
por este Taller. Todo el código de este repositorio es original e independiente.