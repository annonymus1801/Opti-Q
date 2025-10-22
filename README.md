# Opti-Q
Opti-q is na framework to help Question planning.
OptiQ/
│
├── nsga/
│   ├── gnsaga.py                   # Core NSGA-II implementation
│   ├── utils_cost_qoa.py           # Cost and QoA computation utilities
│   ├── perf_eval.py                # Metric normalization and Pareto analysis
│
├── experiments/
│   ├── exp_levels/                 # PerfDB coverage-level experiments (Levels 0–4)
│   ├── exp_operations/             # Impact of max operation count (K=1–5)
│   ├── exp_budgets/                # Budget-sweep experiments (financial/latency constraints)
│   ├── exp_importance/             # LLM importance weighting experiments
│
├── scripts/
│   ├── run_nsga_server.sh          # Batch runner for SLURM or local server
│   ├── collect_results.sh          # Data aggregation from multiple runs
│   ├── preprocess_perfdb.py        # Preprocessing of performance logs
│   ├── analyze_results.ipynb       # Post-experiment visualization and analysis
│
├── data/
│   ├── perfdb.csv                  # Example historical performance records
│   ├── results/                    # NSGA-II output populations and Pareto fronts
│
├── figures/
│   ├── metrics_boxplots.png
│   ├── pareto_fronts.png
│   ├── level_analysis.png
│
├── README.md
└── LICENSE

⚙️ Key Experiments
1. Impact of Historical Data Levels

Simulates availability of prior execution metadata from Level 0 (cold start) → Level 4 (data-rich).

Measures planner adaptability and prediction calibration.

2. Impact of Number of Operations (K)

Varies the maximum number of LLM operations per plan (K = 1–5).

Evaluates expressiveness–efficiency trade-offs.

3. Budget-Constrained Execution

Sweeps financial and latency budgets (F_max, L_max).

Measures degradation in QoA and energy under constrained regimes.

4. LLM Importance / Weighting

Assesses contribution of individual LLMs to Pareto-optimal fronts.

Supports pruning or adaptive inclusion of models.
