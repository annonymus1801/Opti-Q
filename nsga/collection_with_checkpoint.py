import argparse
import json
import logging
import time
from typing import Dict, List, Tuple, Any
import networkx as nx
import numpy as np
import pandas as pd
import torch
import ollama  # type: ignore
import tiktoken
from tqdm import tqdm
import os
import re
import ast
import ollama
# Disable parallel tokenizers to avoid fork issues
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Logging and device setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s â€¢ %(levelname)s â€¢ %(message)s")
logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Token counting
tokenizer = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text or ""))

# LLM mapping
llm_mapping = {
    0: 'mistral:7b',
    1: 'llama3-chatqa:8b',
    2: 'phi4:14b',
    3: 'qwen2.5:14b',
    4: 'gemma3:27b',
    5: 'blend',
}

# =============================================================================
# ENHANCED QUERY CLEANING - REMOVES ALL COT ARTIFACTS
# =============================================================================

def clean_query(text: str) -> str:
    """
    Remove Chain-of-Thought prompting templates from queries.
    Handles multiple formats and removes both prefix and suffix instructions.

    Args:
        text: Raw query text with CoT template

    Returns:
        Cleaned query text with only the essential question and options
    """
    if pd.isna(text) or text == "":
        return text

    text = str(text)

    # Step 1: Remove CoT template at beginning (if exists)
    if "Question:" in text:
        text = re.sub(r"(?s)^.*?(?=Question:)", "", text)

    # Step 2: Remove trailing instructions and templates
    patterns_to_remove = [
        # Original pattern from your code
        r"(?s)Query:[\s\S]*?Only write the final answer without any additional explanation\.\s*",

        # Common trailing instruction
        r"\n\nAbove are multiple-choice questions.*?Simply respond with.*?\.",

        # Other instruction patterns
        r"\n\nInstruction:.*?(?=\n[A-Z]\)|$)",
        r"\n\nOnly write the final answer.*?$",
        r"\n\nSimply respond with.*?$",

        # CoT reasoning templates
        r"\n\nIdentify the core query:.*?$",
        r"\n\nRecall known facts:.*?$",
        r"\n\nEliminate incorrect answers:.*?$",
        r"\n\nConfirm the best answer:.*?$",
        r"\n\nConclude with the final answer:.*?$",

        # Additional common patterns
        r"\n\nThink step by step.*?$",
        r"\n\nProvide your reasoning.*?$",
        r"\n\nExplain your answer.*?$",
        r"\n\nShow your work.*?$"
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    # Step 3: Clean up "Question:" prefix if it exists
    if text.strip().startswith("Question:"):
        text = text.strip()[9:].strip()

    return text.strip()

# =============================================================================
# ULTRA-STRICT PROMPTS - ADDRESSING ALL IDENTIFIED GAPS
# =============================================================================

class UltraStrictPrompts:
    """Ultra-strict prompts designed to eliminate all failure patterns identified in data analysis"""

    # BLEND PROMPTS - CRITICAL FOR SIMILARITY IMPROVEMENT
    BLEND_MMLU = """Question: {question}

Candidate responses:
{candidates}

Task: Select the correct answer from candidates.

Rules:
- Choose the factually accurate option
- If candidates disagree, select the correct one
- If multiple are correct, pick any one
- Use EXACT format below

Required format: "the answer is X)Option text."
Example: "the answer is B)Diabetes."

Output (nothing else):"""

    BLEND_SIMPLEQA = """Question: {question}

Candidate responses:
{candidates}

Task: Provide the most accurate answer.

Rules:
- Select the factually correct response
- If responses disagree, choose the accurate one
- Be direct and concise

Output (nothing else):"""

    # VERIFICATION PROMPTS - STRONG FORMAT CHECKING
    VERIFICATION_MMLU = """Question: {question}

Answer to verify: {context}

Task: Check accuracy and format.

Rules:
- If correct and properly formatted, output it exactly
- If wrong answer, provide correct answer
- Use format: "the answer is X)Option text."

Output (nothing else):"""

    VERIFICATION_SIMPLEQA = """Question: {question}

Answer to verify: {context}

Task: Verify accuracy.

Rules:
- If correct, output it exactly as given
- If wrong, provide correct answer
- Be concise and direct

Output (nothing else):"""

    # LEAF PROMPTS - SIMPLE AND DIRECT
    LEAF_MMLU = """Question: {question}

Select the correct answer.

Format: "the answer is X)Option text."
Example: "the answer is C)Hypertension."

Output (nothing else):"""

    LEAF_SIMPLEQA = """Question: {question}

Provide the accurate answer directly.

Output (nothing else):"""

# =============================================================================
# ULTRA-STRICT POST-PROCESSING - AGGRESSIVE FORMAT ENFORCEMENT
# =============================================================================

def ultra_strict_post_process(output: str, question_type: str) -> str:
    """Aggressive post-processing to enforce exact format matching annotated answers"""

    if not output:
        return ""

    output = output.strip()

    if question_type == 'mmlu':
        # Force lowercase "the answer is" (CRITICAL for similarity)
        if output.startswith("The answer is"):
            output = "the" + output[3:]

        # Extract just the answer part, remove everything after first sentence
        pattern = r'(the answer is [A-E]\)[^.]*\.?)'
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            answer = match.group(1)
            # Ensure it ends with period
            if not answer.endswith('.'):
                answer += '.'
            # Force lowercase start
            if answer.startswith('The'):
                answer = 'the' + answer[3:]
            return answer

        # Fallback: try to extract option and format correctly
        option_pattern = r'([A-E]\)[^.]*)'
        option_match = re.search(option_pattern, output)
        if option_match:
            option_text = option_match.group(1)
            if not option_text.endswith('.'):
                option_text += '.'
            return f"the answer is {option_text}"

    else:  # SimpleQA
        # Remove explanation patterns aggressively
        explanation_cuts = [
            r'\n\nExplanation:.*$',
            r'\. This .*$',
            r'\. The reasoning.*$',
            r'\. Given .*$',
            r'\. Both .*$',
            r'\. However.*$',
            r'\. Therefore.*$',
            r'\. Based on.*$'
        ]

        for pattern in explanation_cuts:
            output = re.sub(pattern, '', output, flags=re.DOTALL | re.IGNORECASE)

        # Take only first sentence for SimpleQA if it's too long
        if len(output) > 100:
            sentences = output.split('.')
            if sentences and sentences[0].strip():
                return sentences[0].strip() + '.'

    return output

# =============================================================================
# ENHANCED DAG EXECUTOR WITH RETRY MECHANISM
# =============================================================================

class UltraStrictDAGExecutor:
    """DAG executor with extreme format enforcement and retry mechanism"""
    def __init__(self, edges: List[Tuple[int,int]], node_models: Dict[int,str], fuser_model: str = "gemma3:27b"):
        self.G = nx.DiGraph()
        self.G.add_edges_from(edges)
        self.G.add_nodes_from(node_models.keys())
        self.fuser_model = fuser_model
        for n, m in node_models.items():
            self.G.nodes[n]['model'] = m

    def determine_question_type(self, question: str, topic: str = "") -> str:
        """Determine if question is MMLU-style or SimpleQA-style"""
        question_lower = question.lower()

        # Check for explicit MMLU indicators
        if topic.endswith('_mmlu'):
            return 'mmlu'

        # Check for multiple choice indicators
        if any(indicator in question for indicator in ['A)', 'B)', 'C)', 'D)', 'E)']):
            return 'mmlu'

        # Check for typical multiple choice patterns
        if 'which of the following' in question_lower or 'select the' in question_lower:
            return 'mmlu'

        return 'simpleqa'

    def format_candidates_minimal(self, candidates: List[str]) -> str:
        """Format candidate responses minimally"""
        if not candidates:
            return "No candidates available."

        formatted = []
        for i, candidate in enumerate(candidates, 1):
            clean_candidate = candidate.strip()
            if not clean_candidate:
                clean_candidate = "[Empty response]"
            formatted.append(f"{i}. {clean_candidate}")

        return "\n".join(formatted)

    def execute_with_validation(self, prompt: str, model: str, question_type: str, max_retries: int = 2):
        """Execute with format validation and retries if needed"""

        for attempt in range(max_retries + 1):
            resp = ollama.generate(model=model, prompt=prompt, options={"device": device})
            raw_output = resp.get('response', '').strip()

            # Apply ultra-strict post-processing
            processed_output = ultra_strict_post_process(raw_output, question_type)

            # Validate format for MMLU
            if question_type == 'mmlu':
                if processed_output.startswith('the answer is') and ')' in processed_output and processed_output.endswith('.'):
                    return processed_output, raw_output
                elif attempt < max_retries:
                    # Retry with even more explicit format enforcement
                    prompt += f"\n\nCRITICAL: Must be exactly 'the answer is X)Option.' format. Example: 'the answer is A)Diabetes.'"
                    continue
            else:
                # For SimpleQA, accept if it's reasonably concise
                if len(processed_output) < 200 and not any(word in processed_output.lower() for word in ['explanation', 'reasoning', 'analysis', 'therefore', 'however']):
                    return processed_output, raw_output

            # If we reach max retries, return what we have
            if attempt == max_retries:
                return processed_output, raw_output

        return processed_output, raw_output

    def build_prompt(self, node_type: str, question: str, candidates: List[str] = None, context: str = None, topic: str = "") -> str:
        """Build appropriate prompt based on node type"""
        question_type = self.determine_question_type(question, topic)
        clean_question = clean_query(question)

        if node_type == 'blend':
            candidates_block = self.format_candidates_minimal(candidates or [])
            if question_type == 'mmlu':
                return UltraStrictPrompts.BLEND_MMLU.format(
                    question=clean_question,
                    candidates=candidates_block
                )
            else:
                return UltraStrictPrompts.BLEND_SIMPLEQA.format(
                    question=clean_question,
                    candidates=candidates_block
                )

        elif node_type == 'verification':
            if question_type == 'mmlu':
                return UltraStrictPrompts.VERIFICATION_MMLU.format(
                    question=clean_question,
                    context=context or ""
                )
            else:
                return UltraStrictPrompts.VERIFICATION_SIMPLEQA.format(
                    question=clean_question,
                    context=context or ""
                )

        elif node_type == 'leaf':
            if question_type == 'mmlu':
                return UltraStrictPrompts.LEAF_MMLU.format(question=clean_question)
            else:
                return UltraStrictPrompts.LEAF_SIMPLEQA.format(question=clean_question)

        return ""

    def execute(self, query: str, topic: str = "") -> Tuple[Dict[int,str], List[Dict[str,Any]]]:
        """Execute the DAG with ultra-strict prompting and validation"""
        outputs: Dict[int, str] = {}
        logs: List[Dict[str,Any]] = []
        question_type = self.determine_question_type(query, topic)

        for node in nx.topological_sort(self.G):
            parents = list(self.G.predecessors(node))
            model = self.G.nodes[node]['model']

            # BLENDING NODES (Multiple parents or explicit 'blend')
            if len(parents) > 1 or model.lower() == 'blend':
                candidate_texts = [outputs[p] for p in parents]
                prompt = self.build_prompt('blend', query, candidates=candidate_texts, topic=topic)

                in_toks = count_tokens(prompt)
                logger.info(f"Blend Node {node}: fuser_model={self.fuser_model}, input_tokens={in_toks}")

                processed_output, raw_output = self.execute_with_validation(
                    prompt, self.fuser_model, question_type
                )
                out_toks = count_tokens(processed_output)

                outputs[node] = processed_output
                logs.append({
                    'node': node,
                    'type': 'blend',
                    'model': self.fuser_model,
                    'prompt': prompt,
                    'candidates': candidate_texts,
                    'raw_output': raw_output,
                    'output': processed_output,
                    'input_tokens': in_toks,
                    'output_tokens': out_toks
                })
                continue

            # VERIFICATION NODES (Single parent)
            elif len(parents) == 1:
                context = outputs[parents[0]]
                prompt = self.build_prompt('verification', query, context=context, topic=topic)

                in_toks = count_tokens(prompt)
                logger.info(f"Verification Node {node}: model={model}, input_tokens={in_toks}")

                processed_output, raw_output = self.execute_with_validation(
                    prompt, model, question_type
                )
                out_toks = count_tokens(processed_output)

                outputs[node] = processed_output
                logs.append({
                    'node': node,
                    'type': 'verification',
                    'model': model,
                    'prompt': prompt,
                    'parent_output': context,
                    'raw_output': raw_output,
                    'output': processed_output,
                    'input_tokens': in_toks,
                    'output_tokens': out_toks
                })

            # LEAF NODES (No parents)
            else:
                prompt = self.build_prompt('leaf', query, topic=topic)

                in_toks = count_tokens(prompt)
                logger.info(f"Leaf Node {node}: model={model}, input_tokens={in_toks}")

                processed_output, raw_output = self.execute_with_validation(
                    prompt, model, question_type
                )
                out_toks = count_tokens(processed_output)

                outputs[node] = processed_output
                logs.append({
                    'node': node,
                    'type': 'leaf',
                    'model': model,
                    'prompt': prompt,
                    'raw_output': raw_output,
                    'output': processed_output,
                    'input_tokens': in_toks,
                    'output_tokens': out_toks
                })

        # Return final sink outputs
        sinks = [n for n in self.G.nodes if self.G.out_degree(n) == 0]
        final = {s: outputs[s] for s in sinks}
        return final, logs

# =============================================================================
# UTILITY FUNCTIONS (Keep existing ones)
# =============================================================================

def get_edge_list(mask: int, k: int, llm_assignment: List[int]) -> List[Any]:
    if k == 1:
        model = llm_mapping[int(llm_assignment[0])]
        return [([0], [model])]
    B = [(mask >> bit) & 1 for bit in range(k * (k - 1) // 2)]
    adj = np.zeros((k, k), dtype=int)
    idx_b = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            adj[i, j] = B[idx_b]; idx_b += 1
    edge_list: List[Tuple[List[int], List[str]]] = []
    for i in range(k):
        for j in range(k):
            if adj[i, j] == 1:
                model_i = llm_mapping[int(llm_assignment[i])]
                model_j = llm_mapping[int(llm_assignment[j])]
                edge_list.append(([i, j], [model_i, model_j]))
    return edge_list

def build_plans_from_csv(df:pd.DataFrame) -> pd.DataFrame:
    # df = pd.read_csv(csv_path)

    # ENHANCED: Clean queries before processing
    if 'Original_Query' in df.columns:
        df['Original_Query'] = df['Original_Query'].apply(clean_query)

    df['plan'] = None
    for idx, row in df.iterrows():
        llm_assignment_list = [x.strip() for x in str(row['assignment']).strip('()').split(',') if x.strip()]
        mask = int(str(row['struct_id']).strip('()').split(',')[0])
        plan = get_edge_list(mask=mask, k=len(llm_assignment_list), llm_assignment=llm_assignment_list)
        df.at[idx, 'plan'] = str(plan)
    return df

# =============================================================================
# ULTRA-STRICT RUNNER FUNCTION
# =============================================================================

def run_plan_queries_ultra_strict(
    df: pd.DataFrame,
    plan_col: str = 'plan',
    query_col: str = 'Original_Query',
    assignment_col: str = 'assignment',
    topic_col: str = 'query_type'
) -> pd.DataFrame:
    """Ultra-strict plan runner focused on exact format matching"""
    results: List[Dict[str,Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing with ultra-strict prompts"):
        plan_raw = row[plan_col]
        plan = ast.literal_eval(plan_raw) if isinstance(plan_raw, str) else plan_raw
        topic = str(row.get(topic_col, "")).strip()

        # Handle plan structure
        if isinstance(plan, list) and plan and all(isinstance(x, int) for x in plan):
            # Single-node plan
            model_str = str(row[assignment_col]).strip('()').split(',')[0].strip()
            model = llm_mapping[int(model_str)] if model_str.isdigit() else model_str
            node_models = {i: model for i in plan}
            edges = []
        else:
            # Multi-node plan
            edges = []
            node_models = {}
            for entry in plan:
                if isinstance(entry[0], list) and len(entry[0]) == 1:
                    # Single-node plan like ([0], ['gemma:2b'])
                    i = entry[0][0]
                    model_i = entry[1][0]
                    node_models[i] = model_i
                else:
                    (i, j), (m_i, m_j) = entry
                    node_models[i] = m_i
                    node_models[j] = m_j
                    edges.append((i, j))

        # Execute with ultra-strict DAG executor
        executor = UltraStrictDAGExecutor(edges, node_models, fuser_model="gemma3:27b")
        final_outs, logs = executor.execute(row[query_col], topic)

        # Prepare result
        out = row.to_dict()
        out.update({
            'Final_Output': ' '.join(final_outs.values()),
            'step_logs': json.dumps(logs, indent=2),
            'num_nodes': len(node_models),
            'num_edges': len(edges),
            'question_type': executor.determine_question_type(row[query_col], topic)
        })
        results.append(out)

    return pd.DataFrame(results)

def run_with_checkpoint(
    df: pd.DataFrame,
    checkpoint_every: int,
    output_path: str
) -> pd.DataFrame:
    results: List[Dict[str,Any]] = []
    total = len(df)
    logger.info(f"Starting collection of {total} rows; checkpoint every {checkpoint_every}")

    for idx, row in tqdm(df.iterrows(), total=total, desc="Collecting"):
        plan = ast.literal_eval(row['plan'])
        # build edges & node_models as in your existing code
        edges, node_models = [], {}
        for pair, models in plan:
            if len(pair) == 1:
                node_models[pair[0]] = models[0]
            else:
                i, j = pair
                node_models[i], node_models[j] = models[0], models[1]
                edges.append((i, j))

        exec = UltraStrictDAGExecutor(edges, node_models, fuser_model="gemma3:27b")
        final_outs, logs = exec.execute(row['Original_Query'], row.get('query_type', ''))
        out = row.to_dict()
        out.update({
            'Final_Output': ' '.join(final_outs.values()),
            'step_logs': json.dumps(logs)
        })
        results.append(out)

        # checkpoint
        if checkpoint_every and (idx + 1) % checkpoint_every == 0:
            chk = f"{os.path.splitext(output_path)[0]}.checkpoint.csv"
            pd.DataFrame(results).to_csv(chk, index=False)
            logger.info(f"[checkpoint @ {idx+1}] wrote {len(results)} rows → {chk}")

    return pd.DataFrame(results)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="UltraStrict collection with checkpointing")
    p.add_argument("-i", "--input-csv",  required=True, help="Input CSV path")
    p.add_argument("-o", "--output-csv", required=True, help="Output CSV path")
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="Checkpoint every N rows")
    args = p.parse_args()

    logger.info(f"Loading input: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    df = build_plans_from_csv(df)

    df_out = run_with_checkpoint(
        df,
        checkpoint_every=args.checkpoint_every,
        output_path=args.output_csv
    )

    logger.info(f"Writing final output to {args.output_csv}")
    df_out.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
