"""
metrics_updated.py — compute Total_Cost / Total_Energy / Total_Latency
from step_logs using split input/output coefficients and blend-by-type.

Assignment mapping:
    0: mistral, 1: llama, 2: phi, 3: qwen, 4: gemma, 5: blend
"""

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s • %(levelname)s • %(message)s"
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Rate tables (example: for query_type = "Art"), keyed by assignment id
# 0: mistral, 1: llama, 2: phi, 3: qwen, 4: gemma, 5: blend
# Values are per token.
# ────────────────────────────────────────────────────────────────────────────────
ART_SPECS: Dict[int, Dict[str, float]] = {
    0: {"input_cost": 2.8e-08,  "output_cost": 5.4e-08,  "input_latency": 0.159983, "output_latency": 0.159983, "input_energy": 0.103939, "output_energy": 0.103939},  # mistral
    1: {"input_cost": 5.5e-08,  "output_cost": 5.5e-08,  "input_latency": 0.15,     "output_latency": 0.15,     "input_energy": 0.10552,  "output_energy": 0.10552},   # llama
    2: {"input_cost": 6e-08,"output_cost": 1.4e-07,  "input_latency": 0.133333, "output_latency": 0.133333, "input_energy": 0.1,      "output_energy": 0.1},       # phi
    3: {"input_cost": 6e-08,"output_cost": 2.4e-07,"input_latency": 0.095852, "output_latency": 0.095852, "input_energy": 0.066938, "output_energy": 0.066938},  # qwen
    4: {"input_cost": 9e-08,"output_cost": 1.6e-07,"input_latency": 0.059915, "output_latency": 0.059915, "input_energy": 0.083433, "output_energy": 0.083433},  # gemma
    5: {"input_cost": 9e-08,"output_cost": 1.6e-07,"input_latency": 0.059915, "output_latency": 0.059915, "input_energy": 0.083433, "output_energy": 0.083433},  # blend
}

def get_rate_table(query_type: str) -> Dict[int, Dict[str, float]]:
    """
    Return the per-assignment rate table for a given query_type.
    Extend this to switch on query_type if you have multiple tables.
    """
    # For now, we return the ART table for everything; customize as needed.
    return ART_SPECS

# ────────────────────────────────────────────────────────────────────────────────
# Assignment mapping (model/type → assignment id)
# ────────────────────────────────────────────────────────────────────────────────
def model_to_assignment(model: str, step_type: str) -> int:
    """
    Map a step's (model, type) to an assignment id.
    Uses 'blend' step_type to identify the blend node.
    """
    t = (step_type or "").lower().strip()
    m = (model or "").lower().strip()
    if t == "blend":
        return 5
    if m.startswith("mistral"):
        return 0
    if m.startswith("llama"):
        return 1
    if m.startswith("phi"):
        return 2
    if m.startswith("qwen"):
        return 3
    if m.startswith("gemma"):
        return 4
    # Fallback: treat as gemma-like
    return 4

# ────────────────────────────────────────────────────────────────────────────────
# Adjacency builder (struct_id bitmask → DAG adjacency)
# ────────────────────────────────────────────────────────────────────────────────
_adj_cache: Dict[Tuple[int, int], np.ndarray] = {}

def build_adjacency(mask: int, k: int) -> np.ndarray:
    """Build adjacency matrix from struct_id mask for k nodes, cached for speed."""
    key = (mask, k)
    if key in _adj_cache:
        return _adj_cache[key]
    B = [(mask >> bit) & 1 for bit in range(k*(k-1)//2)]
    adj = np.zeros((k, k), dtype=int)
    idx = 0
    for i in range(k-1):
        for j in range(i+1, k):
            adj[i, j] = B[idx]
            idx += 1
    _adj_cache[key] = adj
    return adj

# ────────────────────────────────────────────────────────────────────────────────
# Core computation: single pass; split pricing; blend-by-type; longest-path latency
# ────────────────────────────────────────────────────────────────────────────────
def compute_totals_from_step_logs(step_logs_json: str, struct_id: str,
                                  rate_table: Dict[int, Dict[str,float]]) -> Tuple[float,float,float]:
    """
    Given a JSON list of step logs and a struct_id mask, compute:
        Total_Cost, Total_Energy, Total_Latency
    using split input/output pricing and longest-path latency aggregation.
    """
    try:
        steps: List[Dict[str, Any]] = sorted(json.loads(step_logs_json or "[]"),
                                             key=lambda x: x.get("node", 0))
    except json.JSONDecodeError:
        return 0.0, 0.0, 0.0

    k = len(steps)
    mask_str = str(struct_id).strip("()")
    mask = int(mask_str.split(",")[0]) if mask_str else 0
    adj = build_adjacency(mask, k)

    total_cost = 0.0
    total_energy = 0.0
    node_latency = [0.0] * k

    for st in steps:
        a = model_to_assignment(st.get("model",""), st.get("type",""))
        rt = rate_table[a]
        Tin  = float(st.get("input_tokens", 0))
        Tout = float(st.get("output_tokens", 0))
        idx  = int(st.get("node", 0))

        total_cost   += Tin*rt["input_cost"]    + Tout*rt["output_cost"]
        total_energy += Tin*rt["input_energy"]  + Tout*rt["output_energy"]
        node_latency[idx] = Tin*rt["input_latency"] + Tout*rt["output_latency"]

    # longest-path latency on DAG defined by mask
    if k == 0:
        return 0.0, 0.0, 0.0

    longest = node_latency.copy()
    for i in range(k):
        for j in range(i+1, k):
            if adj[i, j]:
                # If edge i->j exists, update the best arrival time at j
                longest[j] = max(longest[j], longest[i] + node_latency[j])

    return total_cost, total_energy, max(longest)

# ────────────────────────────────────────────────────────────────────────────────
# DataFrame helper
# ────────────────────────────────────────────────────────────────────────────────
def add_totals(df: pd.DataFrame, query_type_col: str = "query_type") -> pd.DataFrame:
    """
    Compute Total_Cost / Total_Energy / Total_Latency for each row in df,
    reading per-step tokens from 'step_logs' and the DAG from 'struct_id'.
    """
    out = df.copy()
    totals: List[Tuple[float,float,float]] = []
    for _, r in out.iterrows():
        rate_table = get_rate_table(str(r.get(query_type_col, "Art")))
        c, e, l = compute_totals_from_step_logs(
            r.get("step_logs", "[]"),
            str(r.get("struct_id", "0")),
            rate_table
        )
        totals.append((c, e, l))
    out[["Total_Cost", "Total_Energy", "Total_Latency"]] = pd.DataFrame(totals, index=out.index)
    return out

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compute totals from step logs with split input/output pricing.")
    parser.add_argument("--input", required=True, help="Input CSV (must contain 'step_logs' and 'struct_id').")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--query_type_col", default="query_type", help="Column name for query type (default: query_type).")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df2 = add_totals(df, args.query_type_col)
    df2.to_csv(args.output, index=False)
    logger.info(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
