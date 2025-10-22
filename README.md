# Opti-Q: A Constraint-Based Optimization Framework for Multi-LLM Question Planning

This repository contains the implementation and experimental scripts for the paper:

## 🧩 Repository Structure

```
OptiQ/
│
├── nsga/
│   ├── gnsaga.py                 # Core NSGA-II implementation
│   ├── utils_cost_qoa.py         # Cost and QoA calculation scripts
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
│
├── data/
│   ├── perfdb.csv                # Historical performance metadata
│   ├── results/                  # Output files from NSGA-II runs
│
├── README.md
└── LICENSE
```

---

## 🚀 How to Run

### 1️⃣ Local or SLURM Execution
```bash

```

### 2️⃣ Collect Results
```bash

```


Install all dependencies:
```bash
pip install -r requirements.txt
```
---


