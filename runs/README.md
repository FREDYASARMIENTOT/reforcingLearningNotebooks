# Taller 1 — Dynamic Programming on MilanTaxi

Repositorio de experimentos y resultados del taller.

## Contents

- `experiment_results_<timestamp>.json` — Resultados completos de los 4 experimentos principales.
- `experiment_summary_<timestamp>.csv` — Tabla resumen (CSV).
- `result_<variant>_<algorithm>.json` — Resultados individuales.
- `gamma_experiment.json` — Experimento de efecto de gamma.
- `vi_convergence.png` — Gráfica de convergencia de Value Iteration.

## How to Regenerate

```bash
cd D:\ReforcingLearning\Taller1-Dynamic-Programming
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -c "from rl_project.experiment import run_all_experiments; run_all_experiments()"
```

## Configuration

- **Seed (evaluation):** 42
- **Discount factor γ:** 0.99
- **Tolerance:** 1e-10
- **Slip probability (stochastic):** 0.1
- **Evaluation runs:** 1000 episodes per policy