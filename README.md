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
├── data/
│   ├── perfdb.csv                # Historical performance metadata
│  
│
├── README.md
└── LICENSE
```



Install all dependencies:
```bash
pip install -r requirements.txt
```
---


