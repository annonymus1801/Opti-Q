#!/usr/bin/env python3
import argparse
import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import ollama  # type: ignore
import tiktoken
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ────────────────────────────────────────────────────────────────────────────────
# Logging and device
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s • %(levelname)s • %(message)s")
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────────────────────
# Load sentence-transformers model
# ────────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ────────────────────────────────────────────────────────────────────────────────
# Helpers for embeddings
# ────────────────────────────────────────────────────────────────────────────────
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def chunk_text(text: str, chunk_size: int = 250) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embeddings_for_large_text(text: str, chunk_size: int = 250) -> torch.Tensor:
    texts = chunk_text(text, chunk_size) if len(text.split()) > chunk_size else [text]
    embeddings = []
    for chunk in texts:
        encoded = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        pooled = mean_pooling(output, encoded["attention_mask"])
        normalized = F.normalize(pooled, p=2, dim=1)
        embeddings.append(normalized)
    return torch.mean(torch.stack(embeddings), dim=0)

def safe_str(obj: Any) -> str:
    if isinstance(obj, (str, int, float)):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if pd.isna(obj):
        return ""
    return str(obj)

# ────────────────────────────────────────────────────────────────────────────────
# QoA computation
# ────────────────────────────────────────────────────────────────────────────────
def compute_qoa(annotated: str, llm_ans: str) -> float:
    if isinstance(annotated, str) and isinstance(llm_ans, str):
        if annotated.strip() and annotated in llm_ans:
            return 1.0
    try:
        emb_ann = get_embeddings_for_large_text(safe_str(annotated))
        emb_llm = get_embeddings_for_large_text(safe_str(llm_ans))
        return float(cosine_similarity(
            emb_ann.cpu().numpy().reshape(1,-1),
            emb_llm.cpu().numpy().reshape(1,-1)
        )[0][0])
    except Exception as e:
        logger.error(f"Error computing QoA: {e}")
        return np.nan

# ────────────────────────────────────────────────────────────────────────────────
# DataFrame processing
# ────────────────────────────────────────────────────────────────────────────────
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Computing Quality_of_Answer for {len(df)} rows")
    tqdm.pandas()
    df["Quality_of_Answer"] = df.progress_apply(
        lambda r: compute_qoa(r.get("Annotated Answer"), r.get("Final_Output")),
        axis=1,
    )
    return df

# ────────────────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compute QoA for LLM outputs")
    parser.add_argument(
        "-i", "--input-file", required=True, help="Path to input CSV"
    )
    parser.add_argument(
        "-o", "--output-file", required=True, help="Path to save output CSV"
    )
    args = parser.parse_args()

    logger.info(f"Loading {args.input_file}")
    df = pd.read_csv(args.input_file, dtype=str)

    df = process_dataframe(df)

    logger.info(f"Saving results to {args.output_file}")
    df.to_csv(args.output_file, index=False)

    logger.info("QoA summary:\n" + df["Quality_of_Answer"].describe().to_string())
    logger.info("Done.")

if __name__ == "__main__":
    main()

