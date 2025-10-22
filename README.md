# Opti-Q: A Constraint-Based Optimization Framework for Multi-LLM Question Planning

This repository contains the implementation and experimental scripts for the paper:

> **Opti-Q: A Constraint-Based Optimization Framework for Multi-LLM Question Planning **  


---

## 📘 Overview

**Opti-Q** introduces a multi-objective optimization framework that plans and executes Large Language Model (LLM) questions under cost, latency, and energy constraints while maximizing **Quality of Answer (QoA)**.  
The system uses the **NSGA-II genetic algorithm** to find Pareto-optimal execution plans for question-answering workflows.

---

## 🧩 Repository Structure

```
OptiQ/
│
├── nsga/
│   ├── gnsaga.py                 # Core NSGA-II implementation
│   ├── utils_cost_qoa.py         # Cost and QoA calculation scripts
│   ├── perf_eval.py              # Performance normalization and Pareto analysis
│
├── experiments/
│   ├── exp_levels/               # PerfDB level experiments (0–4)
│   ├── exp_operations/           # Impact of max operation count (K=1–5)
│   ├── exp_budgets/              # Budget-constrained runs (financial/latency)
│   ├── exp_importance/           # LLM importance and weighting experiments
│
├── scripts/
│   ├── run_nsga_server.sh        # Bash script to run experiments on server/SLURM
│   ├── collect_results.sh        # Script to aggregate experiment outputs
│   ├── preprocess_perfdb.py      # Preprocessing for performance database
│   ├── analyze_results.ipynb     # Post-experiment plots and analysis
│
├── data/
│   ├── perfdb.csv                # Historical performance metadata (sample)
│   ├── results/                  # Output files from NSGA-II runs
│
├── figures/
│   ├── metrics_boxplots.png
│   ├── pareto_fronts.png
│   ├── level_analysis.png
│
├── README.md
└── LICENSE
```

---

## ⚙️ Experiments Included

| Experiment | Description |
|-------------|-------------|
| **Level Experiments (0–4)** | Tests planner adaptability with increasing PerfDB historical coverage. |
| **Max Operations (K)** | Evaluates how the number of operations affects expressiveness and efficiency. |
| **Budget Constraints** | Sweeps over financial and latency limits to study trade-offs. |
| **LLM Importance** | Measures contribution and sensitivity of individual LLMs to Pareto fronts. |

Each experiment reports **QoA, Cost, Latency, and Energy** with **95% confidence intervals** over multiple runs.

---

## 🚀 How to Run

### 1️⃣ Local or SLURM Execution
```bash

```

### 2️⃣ Collect Results
```bash

```


---

## 🧮 Core Components

| File | Function |
|------|-----------|
| `gnsaga.py` | NSGA-II algorithm implementation (elitism, crossover, mutation) |
| `utils_cost_qoa.py` | Cost and QoA computation functions |
| `perf_eval.py` | Normalization, Pareto front extraction, CI calculation |
| `run_nsga_server.sh` | SLURM-compatible script for running experiments |
| `collect_results.sh` | Collects and merges distributed result files |

---

## 📊 Metrics

| Metric | Description |
|---------|-------------|
| **QoA** | Quality of Answer (using Mini-LV6 or exact match) |
| **Cost** | Model invocation cost (USD × 10⁻⁴ scale) |
| **Latency** | Execution time in seconds |
| **Energy** | Estimated consumption (Joules) |

---

## 🧠 Methodology

The framework optimizes four competing objectives using **NSGA-II**:
1. Maximize QoA  
2. Minimize Cost  
3. Minimize Latency  
4. Minimize Energy  

Each population represents a multi-LLM execution plan, and evolution proceeds via crossover, mutation, and Pareto dominance sorting.

---

## 🧰 Requirements

- Python ≥ 3.9  
- NumPy  
- Pandas  
- Matplotlib  
- SciPy  
- tqdm  
- jsonlines  

Install all dependencies:
```bash
pip install -r requirements.txt
```
---

## 📜 License

Released under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.

---

