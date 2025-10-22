# -*- coding: utf-8 -*-
import numpy as np
import math, copy
from difflib import SequenceMatcher
import Levenshtein
import time
from tqdm import tqdm
import pandas as pd
import itertools
import random, csv
from typing import Dict, Tuple, List
import networkx as nx
import json
import ast
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import networkx as nx
import pandas as pd
from pprint import pprint
import argparse
import json
import copy



 # default hyperparameters (can be overridden via config file)
# ... none currently
REPETITIONS = 5
RESULTS_FILE = "nsga_results.csv"
POP_SIZE       = 50
GENERATIONS    = 100
MAX_NODES      = 5
# probabilities used in `mutate`
ADD_NODE_PROB       = 0.1
FLIP_EDGE_PROB      = 0.3
MODEL_MUTATION_PROB = 0.2

# default query‐types
QUERY_TYPES = [
    'Art', 'Geography', 'History', 'Music', 'Other', 'Politics',
    'Science and technology', 'Sports', 'TV shows', 'Video games',
    'biology_mmlu', 'business_mmlu', 'chemistry_mmlu', 'computer science_mmlu',
    'economics_mmlu', 'engineering_mmlu', 'health_mmlu', 'history_mmlu',
    'law_mmlu', 'math_mmlu', 'other_mmlu', 'philosophy_mmlu',
    'physics_mmlu', 'psychology_mmlu'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run NSGA-II with config file")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file")
    return parser.parse_args()

def load_config(path):
  with open(path) as f:
      cfg = json.load(f)     
  # (no overrides previously)
  global REPETITIONS, RESULTS_FILE, POP_SIZE, GENERATIONS, MAX_NODES
  global ADD_NODE_PROB, FLIP_EDGE_PROB, MODEL_MUTATION_PROB, QUERY_TYPES
  REPETITIONS = cfg.get("repetitions", REPETITIONS)
  RESULTS_FILE = cfg.get("results_file_name", RESULTS_FILE)
  POP_SIZE       = cfg.get("pop_size", POP_SIZE)
  GENERATIONS    = cfg.get("generations", GENERATIONS)
  MAX_NODES      = cfg.get("max_nodes", MAX_NODES)
  ADD_NODE_PROB       = cfg.get("prob_add_node", ADD_NODE_PROB)
  FLIP_EDGE_PROB      = cfg.get("prob_flip_edge", FLIP_EDGE_PROB)
  MODEL_MUTATION_PROB = cfg.get("prob_model_mutation", MODEL_MUTATION_PROB)
  QUERY_TYPES = list(cfg["query_types"])

import itertools

def canonical_representation(adj_matrix, k):
    """Compute highest-value adjacency bitmask among all isomorphic labelings of a DAG."""
    nodes = list(range(k))
    best_mask = -1
    for perm in itertools.permutations(nodes):
        # Check if perm is a valid topological order (no edge goes from later to earlier in this perm)
        valid_topo = True
        pos = {node: idx for idx, node in enumerate(perm)}
        for i in range(k):
            for j in range(k):
                if adj_matrix[i][j] == 1 and pos[i] > pos[j]:
                    valid_topo = False
                    break
            if not valid_topo:
                break
        if not valid_topo:
            continue
        # Build adjacency in new labeling and compute bitmask
        new_mask = 0
        bit_index = 0
        for a in range(k-1):
            for b in range(a+1, k):
                # Map original nodes corresponding to new labels a, b
                # Find original nodes x,y such that perm.index(x)=a, perm.index(y)=b
                x = perm[a]; y = perm[b]
                if adj_matrix[x][y] == 1:
                    new_mask |= (1 << bit_index)
                bit_index += 1
        if new_mask > best_mask:
            best_mask = new_mask
    return best_mask

def generate_nonisomorphic_dags(max_nodes):
    """Generate all fully-connected DAGs (up to max_nodes) with one sink, returning unique canonical structures."""
    unique_dags = []
    seen_structs = set()
    for k in range(1, max_nodes+1):
        num_edges = k*(k-1)//2  # number of possible edges in upper triangle
        for mask in range(1 << num_edges):
            # Skip empty graph for k>1 (not fully connected)
            if k > 1 and mask == 0:
                continue
            # Reconstruct adjacency matrix from bitmask
            adj = [[0]*k for _ in range(k)]
            bit_index = 0
            for i in range(k-1):
                for j in range(i+1, k):
                    if mask & (1 << bit_index):
                        adj[i][j] = 1
                    bit_index += 1
            # Check one-sink condition: only the highest-index node has out-degree 0
            outdeg = [0]*k; indeg = [0]*k
            for i in range(k):
                for j in range(k):
                    if adj[i][j] == 1:
                        outdeg[i] += 1
                        indeg[j] += 1
            # Highest node index = k-1 is always sink (no outgoing by construction); ensure no other sink
            if any(outdeg[node] == 0 for node in range(k-1)):
                continue
            # Check full connectivity (every node can reach the sink)
            reachable = [False]*k
            reachable[-1] = True  # mark sink
            # BFS backward from sink
            queue = [k-1]
            while queue:
                cur = queue.pop(0)
                for prev in range(cur):
                    if adj[prev][cur] == 1 and not reachable[prev]:
                        reachable[prev] = True
                        queue.append(prev)
            if not all(reachable):
                continue
            # Compute canonical structure id for this DAG
            canon_mask = canonical_representation(adj, k)
            if (k, canon_mask) not in seen_structs:
                print(f"k={k}, mask={canon_mask}, adj = {adj}")
                seen_structs.add((k, canon_mask))
                unique_dags.append((k, canon_mask))
    return unique_dags

def assign_models_to_dag(k, struct_id):
    # Reconstruct adjacency to get in-degrees
    def adjacency_from_mask(k, mask):
        adj_list = {i: [] for i in range(k)}
        bit_index = 0
        for i in range(k-1):
            for j in range(i+1, k):
                if mask & (1 << bit_index):
                    adj_list[i].append(j)
                bit_index += 1
        return adj_list
    def indegree_list(k, adj_list):
        indeg = [0]*k
        for u in adj_list:
            for v in adj_list[u]:
                indeg[v] += 1
        return indeg

    adj_list = adjacency_from_mask(k, struct_id)
    indeg = indegree_list(k, adj_list)
    assignment = []
    for node in range(k):
        if indeg[node] > 1:
            assignment.append(5)         # blending model for merge nodes
        else:
            assignment.append(random.randint(0, 4))  # random base LLM for others
    return assignment


def adjacency_matrix_from_mask(mask, k):
    """Generates an adjacency matrix from a given mask and number of nodes.
    Args:
        mask: An integer representing the connections in the graph.
        k: The number of nodes in the graph.
    Returns:
        A list of lists representing the adjacency matrix.
    """
    adj_matrix = [[0] * k for _ in range(k)]
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if (mask >> bit_index) & 1:
                adj_matrix[i][j] = 1
            bit_index += 1
    return adj_matrix

def evaluate_individual_V2(struct_id, assignment, query_type, query_tokens, blending_prompt_tokens, ctx_tokens,
                           df_history):
    print(f'Evaluating individual structure_id: {struct_id} assignment: {assignment}')
    k = len(assignment)
    # Reconstruct adjacency list from struct_id
    adj_list = {i: [] for i in range(k)}
    bit_index = 0
    for i in range(k-1):
        for j in range(i+1, k):
            if struct_id & (1 << bit_index):
                adj_list[i].append(j)
            bit_index += 1

    llm_assignment = []
    for model in assignment:
      llm_assignment.append(str(model))

    # print('adj list')
    # pprint(adj_list)


    adjacency_matrix = adjacency_matrix_from_mask(struct_id, k)
    matrix = np.array(adjacency_matrix)
    size = matrix.shape[0]
    # print('matrix', adjacency_matrix)

    # Create directed graph
    G = nx.DiGraph()
    for idx, node_dict in enumerate(llm_assignment):
      G.add_node(idx)
      G.nodes[idx]['info'] = node_dict

    # Add edges based on the upper triangular adjacency matrix
    for i in range(size):
        for j in range(i + 1, size):
            if matrix[i][j] == 1:
                # print(f'addding edge {i} to {j}')
                G.add_edge(i, j)


    # print('adj matrix')
    # pprint(adjacency_matrix)
    # G = nx.DiGraph()
    # G.add_nodes_from(list(range(k)))
    # for key in adj_list:
    #   for j in range(len(adj_list[key])):
    #     G.add_edge(key, adj_list[key][j])

    # print('in eval indiv')
    # for edge in G.edges:
    #   print('ei edge', edge)
    # print("Passing:", struct_id, assignment)
    # print('plotting graph', list(G.nodes()))
    # if len(list(G.edges)) == 0:
    #   print('no edges')
    # plot_graph(G)
    # print('graph plotted', list(G.nodes()))


    final_metrics = estimate_schedule_v3(
      G,
      llm_assignment,
      query_type,
      query_tokens,
      blending_prompt_tokens,
      ctx_tokens,
      df_history,
      levenshtein_threshold=0.70,
      turn_off_exact_fuzzy_matching=False
    )
    # global_debug_graph.append((struct_id, assignment))
    return (final_metrics.final_cost, final_metrics.final_latency, final_metrics.final_energy, final_metrics.quality_of_answer)

import json
import ast
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import networkx as nx
import pandas as pd
from pprint import pprint
import itertools


## --- newly added
def get_adj_from_graph(G: nx.DiGraph):
  node_order = sorted(G.nodes())
  adj_matrix = nx.to_numpy_array(G, nodelist=node_order, dtype=int)
  return adj_matrix

## --- newly added
def canonical_label_for_isomorphism(G: nx.DiGraph):
  adj_matrix = get_adj_from_graph(G)
  return canonical_representation(adj_matrix, len(G.nodes()))


def construct_subdag_behind_node(
  whole_graph: nx.DiGraph,
  sink_node: int,
  llm_assignment: List[str]
):
  """
  The idea here is to construct an entire graph behind a single node
  - We know that we are dealing with DAGs, so there are no cycles
  - given some @sink_node in @whole_graph, we can construct a subgraph
    where every node and the subgraph's connections lead up to the @sink_node

  Approach:
  - Iterative DFS where we start with @sink_node, and add @sink_node's predecessors to
    subgraph, then the process is repeated by adding the predecessors's predecessors to the graph
  - Effectively, build the subgraph
  """
  # stack to emulate recursion
  stack = [sink_node]
  subgraph = nx.DiGraph()
  seen = set()
  while stack:
    node = stack.pop(-1)
    if node in seen:
      continue
    subgraph.add_node(node)
    seen.add(node)

    predecessors = list(whole_graph.predecessors(node))
    for pred in predecessors:
      subgraph.add_edge(pred, node)
      stack.append(pred)

  # Get the proper llm assignment by using the node values as an index into the llm_assignment list
  subgraph_llm_assignment = [llm_assignment[node] for node in sorted(subgraph.nodes())]
  return (subgraph, subgraph_llm_assignment, sink_node)


def get_sub_llm_assignment(subDAG, llm_assignment):
  return [llm_assignment[node] for node in sorted(subDAG.nodes())]

def get_sub_llm_assignment_given_nodes(nodes, llm_assignment):
  return [llm_assignment[node] for node in sorted(nodes)]

def is_blend_node(G: nx.DiGraph, node: int) -> bool:
  return G.in_degree(node) >= 2

def is_sequential_node(G: nx.DiGraph, node: int) -> bool:
  return G.in_degree(node) == 1

def is_start_node(G: nx.DiGraph, node: int) -> bool:
  return G.in_degree(node) == 0

def is_sink(G1: nx.DiGraph, G2: nx.DiGraph, node: int) -> bool:
  return G1.out_degree(node) == 0

@dataclass
class LLMMetrics:
    input_cost: float=0.0
    input_latency: float=0.0
    input_energy: float=0.0
    output_cost: float=0.0
    output_latency: float=0.0
    output_energy: float=0.0
    quality_of_answer: float=0.0
    average_output_tokens: float=0.0
    final_cost: float=0.0
    final_latency: float=0.0
    final_energy: float=0.0

@dataclass
class BFSNode:
    LLM_name:str
    node: int
    metrics: LLMMetrics # e.g., cost, latency, energy, qoa
    accumulated_metrics: LLMMetrics  # Accumulated along the path
    first_node_in_seq: bool
    def __iter__(self):
      return iter((self.LLM_name, self.node, self.metrics, self.accumulated_metrics, self.first_node_in_seq))

@dataclass
class ProcessingTableEntry:
    LLM: str
    node: int
    metrics: LLMMetrics
    accumulated_metrics: LLMMetrics

@dataclass
class BlendingNodeDBReference:
    inputs: Dict[str, float]
    output: float =  0.0


def special_get_subdag_metrics_for_one_blend_operations(
  subdag: nx.DiGraph,
  sub_assignment: List[str],
  query_type: str,
  df_history: pd.DataFrame
):
  """
  Improved version with proper structure_id comparison
  """
  adj_matrix = get_adj_from_graph(subdag) # construct
  structure_id = canonical_representation(adj_matrix, len(sub_assignment))
  # print('sub assignment', sub_assignment)
  assignment_str = ",".join(sub_assignment)


  all_valid_forms = []
  llms = sub_assignment[0:-1]
  perms = set(itertools.permutations(llms))  # remove duplicates
  for perm in perms:
    all_valid_forms.append(",".join(list(perm) + [sub_assignment[-1]]))

  # print('all valid forms:')
  # pprint(all_valid_forms)

  possible = df_history[
    (df_history["structure_id"] == structure_id) &
    (df_history["llm_assignments"].isin(all_valid_forms)) &
    (df_history["query_type"] == query_type)
  ]

  if len(possible) == 0:
    return None
  else:
    ### final_cost, final_latency, final_energy will be used for matched subschedules
    ### fields like input_cost and output_cost are used for estimation purposes
    assert len(possible) == 1
    return LLMMetrics(
      input_cost=possible["input_cost"].mean(),
      input_latency=possible["input_latency"].mean(),
      input_energy=possible["input_energy"].mean(),
      output_cost=possible["output_cost"].mean(),
      output_latency=possible["output_latency"].mean(),
      output_energy=possible["output_energy"].mean(),
      quality_of_answer=possible["qoa"].mean(),
      average_output_tokens=possible["average_output_tokens"].mean(),
      final_cost=possible["cost"].mean(),
      final_latency=possible["latency"].mean(),
      final_energy=possible["energy"].mean(),
    )


def get_subdag_metrics_v7(
  subdag: nx.DiGraph,
  sub_assignment: List[str],
  query_type: str,
  df_history: pd.DataFrame
) -> Tuple[float, float, float, float, float]:
  """
  Improved version with proper structure_id comparison
  """
  adj_matrix = get_adj_from_graph(subdag) # construct
  structure_id = canonical_representation(adj_matrix, len(sub_assignment))
  # print('sub assignment', sub_assignment)
  assignment_str = ",".join(sub_assignment)
  # print("structure id for match", structure_id)
  # print("assignment str for match", assignment_str)

  possible = df_history[
    (df_history["structure_id"] == structure_id) &
    (df_history["llm_assignments"] == assignment_str) &
    (df_history["query_type"] == query_type)
  ]

  if len(possible) == 0:
    return None
  else:
    ### final_cost, final_latency, final_energy will be used for matched subschedules
    ### fields like input_cost and output_cost are used for estimation purposes
    return LLMMetrics(
      input_cost=possible["input_cost"].mean(),
      input_latency=possible["input_latency"].mean(),
      input_energy=possible["input_energy"].mean(),
      output_cost=possible["output_cost"].mean(),
      output_latency=possible["output_latency"].mean(),
      output_energy=possible["output_energy"].mean(),
      quality_of_answer=possible["qoa"].mean(),
      average_output_tokens=possible["average_output_tokens"].mean(),
      final_cost=possible["cost"].mean(),
      final_latency=possible["latency"].mean(),
      final_energy=possible["energy"].mean(),
    )

def get_single_model_metrics(llm_assignment: List[str], query_type:str, df_history: pd.DataFrame):
  single_models = {}
  for llm in llm_assignment:
    single_model_DAG = nx.DiGraph()
    single_model_DAG.add_node(llm)
    single_model_metrics = get_subdag_metrics_v7(
      single_model_DAG,
      [llm],
      query_type,
      df_history
    )
    if single_model_metrics is not None:
      single_models[llm] = single_model_metrics
  return single_models


def get_pairwise_metrics(llm_assignment: List[str], query_type:str, df_history: pd.DataFrame):
  pairwise_models = {}
  pairwise_DAG = nx.DiGraph()
  pairwise_DAG.add_nodes_from([0, 1])
  pairwise_DAG.add_edge(0, 1)
  for i in range(len(llm_assignment)):
    for j in range(len(llm_assignment)):
      if llm_assignment[i] != 5 or llm_assignment[j] !=5:
        node_i, node_j = llm_assignment[i], llm_assignment[j]
        pair_llm_assignment = [node_i, node_j]
        pairwise_metrics = get_subdag_metrics_v7(
          pairwise_DAG,
          pair_llm_assignment,
          query_type,
          df_history
        )
      if pairwise_metrics is not None:
        pairwise_models[(node_i, node_j)] = pairwise_metrics
  return pairwise_models

def convert_metrics_to_dict(metrics):
  return {
      "input_cost": metrics[0],
      "input_latency": metrics[1],
      "input_energy":metrics[2],
      "output_cost": metrics[3],
      "output_latency": metrics[4],
      "output_energy": metrics[5],
      "quality_of_answer": metrics[6],
      "average_output_tokens": metrics[7],
      "final_cost": metrics[8],
      "final_latency": metrics[9],
      "final_energy": metrics[10],
    }

# not used currently in estimation code flow
def fallback_estimation_scheme_v2(
    G: nx.DiGraph,
    assignment: List[str],
    query_type: str,
    df_history: pd.DataFrame,
    get_output_as_dict=False,
    print_debug_log=False
):
  debug_log = []
  topo_order = list(nx.topological_sort(G))
  blend_nodes = list(filter(lambda x: G.in_degree(x) > 1, topo_order[::-1]))
  last_blend_node = blend_nodes[0]
  # get all the inputs for the last blend node
  # if one of the inputs itself is another blend node, get inputs for that as well
  inputs = set()
  stack = [last_blend_node]
  while stack:
    node = stack.pop(-1)
    if node in inputs: continue
    if is_blend_node(G, node):
      stack.extend(list(G.predecessors(node)))
    else:
      inputs.add(node)

  debug_log.append(f'All predecessors input of nested blending node search: {inputs}')
  subDAG = nx.DiGraph()
  subDAG.add_nodes_from(list(inputs) + [last_blend_node])
  for node in inputs:
    if node == last_blend_node: continue
    subDAG.add_edge(node, last_blend_node)

  sub_llm_assignment = get_sub_llm_assignment(subDAG, assignment)
  debug_log.append(f'nodes of the new constructed subDAG for parallel-fallback-match: {subDAG.nodes()}')
  debug_log.append(f'structure_id: {canonical_label_for_isomorphism(subDAG)}')
  debug_log.append(f'llm assignment: {sub_llm_assignment}')

  metrics = get_subdag_metrics_v7(subDAG, sub_llm_assignment, query_type, df_history)
  debug_log.append(f'metrics: {metrics}')
  if print_debug_log:
    print('\n'.join(debug_log))
  return metrics, last_blend_node

def sequential_delta(node_A: str, node_B:str, single_model_metrics, pairwise_metrics):
  final = pairwise_metrics[(node_A, node_B)].quality_of_answer
  initial = single_model_metrics[node_B].quality_of_answer # ask aamir if he wants to do the A or B in A->B for inital, but I think B makes more sense here
  delta = (final - initial) / initial
  return delta


def fuzzy_ranked_matches_from_database(
    llm_assignment: List[str],
    target_structure_id: int,
    target_last_node: str,
    query_type: str,
    df_history: pd.DataFrame,
    threshold: float = 0.7,
    w_seq: float = 0.33,
    w_lev: float = 0.33,
    w_jac: float = 0.33,
    bonus_for_last_node: float = 0.01
) -> List[Tuple[List[str], Tuple[float, float, float, float], float]]:
    target_seq_str    = ",".join(llm_assignment)
    target_struct_str = str(target_structure_id)
    # print('target_seq_str',target_seq_str)
    # print('target_struct_str', target_struct_str) # in full form
    matches = []
    best_score = 0.0
    best_row = None
    best_candidate = None

    """
    Note to self:
    cand: list
    cand_str: string representation of cand, but remove brackets and double-quotes
    struct_str: string representation of canonical label
    target_seq_str: string representation of llm_assignment, must be same format at cand_str
    target_struct_str: string representation of canonical label, must be same format at struct_str
    """

    def clean_assignment(s: str) -> List[str]:
      """
      Cleans a string representation of llm_assignments, removing extra chars.
      Returns: A list of node names.
      """
      import re
      # Strip brackets & quotes, split on commas
      inner = s.strip("[]")
      inner = inner.replace("'", "").replace(" ", "")
      inner = re.sub(",+", ",", inner)  # Repeated commas -> single
      inner = inner.lstrip(",") # Remove Leading Comma
      inner = inner.rstrip(",")  # Remove trailing comma
      inner = inner.replace("\"", "")
      return inner

    for _, row in df_history.iterrows():
        # we do not want to match with single or pairwise for our fuzzy matches
        if row["query_type"] != query_type:
            continue
        
        cand_str = row["llm_assignments"]
        struct_str = str(row["structure_id"])
        cand = [item.strip() for item in cand_str.split(',')]

        if struct_str != target_struct_str:
           continue

        if not (cand and cand[-1] == target_last_node):
           continue


        # compute hybrid components
        # print(f'llm_assignment: {llm_assignment}, {type(llm_assignment)}, cand {cand}, {type(cand)}, cand_str: {cand_str}, struct_str: {struct_str} target_struct_str:{target_struct_str}')
        # make sure cand is a list, and cand_str is string representation without bracket and quotes to wrap the LLMs
        seq_sim    = SequenceMatcher(None, llm_assignment, cand).ratio()
        lev_sim    = Levenshtein.ratio(target_seq_str, cand_str)
        jac_sim    = len(set(llm_assignment) & set(cand)) / len(set(llm_assignment) | set(cand)) if llm_assignment else 1.0
        #struct_sim = 1.0 if struct_str == target_struct_str else 0.0
        #last_node_eq = 1.0 if cand and cand[-1] == target_last_node else 0.0
        
        hybrid = w_seq*seq_sim + w_lev*lev_sim + w_jac*jac_sim
        #if cand and cand[-1] == target_last_node:
        #   hybrid += bonus_for_last_node

        if hybrid > best_score:
            best_score     = hybrid
            best_row       = row
            best_candidate = cand_str

    # print('made it here')
    # print('best_score', best_score)
    # print('best_row', best_row)
    # print('best_candidate', best_candidate)

    if best_row is not None and best_score >= threshold:
        return (
            best_candidate,
            (best_row["cost"], best_row["latency"],
             best_row["energy"], best_row["qoa"]),
            best_score
        )
    return (None, None, best_score)

def fuzzy_match_sequential_chain(graph:nx.DiGraph, whole_llm_assignment: List[str], query_type:str, single_model_metrics, df_history: pd.DataFrame, levin_threshold):
  # Assume G is fully sequential
  G = graph.copy()
  llm_assignment = whole_llm_assignment.copy()
  # print('passed in llm_assignment', whole_llm_assignment)
  topo_sort = list(nx.topological_sort(G))
  metrics = None
  last_i = 0
  # print('in fuzzy')
  # print('plot of whole subgraph under match consideration', llm_assignment)
  # plot_graph(G)
  MIN_LENGTH_OF_SEQUENCE_TO_MATCH = 2
  for i, start_node in enumerate(topo_sort):
    # print('matching', llm_assignment)
    # plot_graph(G)
    # print('i',i, 'sn:', start_node)
    # print('llm assignmnet', llm_assignment)
    if len(topo_sort) - i < MIN_LENGTH_OF_SEQUENCE_TO_MATCH:
      break
    # print(start_node, llm_assignment)
    target_last_node = llm_assignment[-1]
    # print('target_last_node', target_last_node)
    target_structure_id = canonical_label_for_isomorphism(G)
    _, match, score = fuzzy_ranked_matches_from_database(
        llm_assignment,
        target_structure_id,
        target_last_node,
        query_type,
        df_history,
    )
    #print('levin_threshold', levin_threshold)
    if match and score > levin_threshold:
      # print('fuzzy match found')
      metrics = LLMMetrics(average_output_tokens=single_model_metrics[target_last_node].average_output_tokens, final_cost=match[0], final_latency=match[1], final_energy=match[2], quality_of_answer=match[3])
      break
    G.remove_node(start_node)
    llm_assignment.pop(0)

  return metrics, G, topo_sort[-1]

# NEW
# calculate the input and output costs of node and its successor, if the connection is sequential
# if node is first in line of sequential chain, then this function will calculate the input and output costs of Node, and then do the same for the successor
# if node is not the first, then all we do is calculate the successors incurred costs, and the calling function will add these costs onto existing metrics
def calculate_cost_sequential_v2(
    query_tokens: int,
    node_A_avg_tokens: int,
    ctx_prompt_tokens: int,
    node_B_avg_tokens: int,
    cost_factors_A: LLMMetrics,
    cost_factors_B: LLMMetrics,
    first_node_in_seq: bool,
):
  if first_node_in_seq:
    # print('in here')
    input_tokens_A = query_tokens
    output_tokens_A = node_A_avg_tokens
    input_tokens_B = ctx_prompt_tokens + node_A_avg_tokens + query_tokens
    output_tokens_B = node_B_avg_tokens
    # print('input tokens_A', input_tokens_A, 'output_tokens_A', output_tokens_A)
    # print('input tokens_B', input_tokens_B, 'output_tokens_B', output_tokens_B)
    # print('input cost factor A', cost_factors_A.input_cost, 'output', cost_factors_A.output_cost, type(cost_factors_A.output_cost) )
    # print('input cost factor B', cost_factors_B.input_cost, 'output', cost_factors_B.output_cost )
    cost = (input_tokens_A * cost_factors_A.input_cost + output_tokens_A * cost_factors_A.output_cost) + (input_tokens_B * cost_factors_B.input_cost + output_tokens_B * cost_factors_B.output_cost)
    latency = (input_tokens_A * cost_factors_A.input_latency + output_tokens_A * cost_factors_A.output_latency) + (input_tokens_B * cost_factors_B.input_latency + output_tokens_B * cost_factors_B.output_latency)
    energy = (input_tokens_A * cost_factors_A.input_energy + output_tokens_A * cost_factors_A.output_energy) + (input_tokens_B * cost_factors_B.input_energy + output_tokens_B * cost_factors_B.output_energy)
    # print('final cost', cost)
    return cost, latency, energy
  else:
    input_tokens_B = ctx_prompt_tokens + node_A_avg_tokens + query_tokens
    output_tokens_B = node_B_avg_tokens
    cost = cost_factors_B.input_cost * input_tokens_B + cost_factors_B.output_cost * output_tokens_B
    energy = cost_factors_B.input_energy * input_tokens_B + cost_factors_B.output_energy * output_tokens_B
    latency = cost_factors_B.input_latency * input_tokens_B + cost_factors_B.output_latency * output_tokens_B
    return cost, latency, energy

def calculate_cost_parallel(
    llm_assignments,
    blend_node,
    current_metrics,
    blending_node_metrics,
    processing_table_entries,
    blending_reference_table,
    single_model_metrics,
    query_tokens,
    blending_prompt_tokens,
    turn_off_blend_table_check=False
):
    # ─── EDIT 1 ───
    # fallback constant for any missing/zero initial QoA
    DEFAULT_QOA = 0.5
    current_metrics = copy.deepcopy(current_metrics)

    # pull out the reference‐table entry
    use_blending_reference_table_entry = True
    if blend_node not in blending_reference_table or turn_off_blend_table_check:
      use_blending_reference_table_entry = False

    # print(use_blending_reference_table_entry, 'be', 'the blend node', blend_node)

    blending_reference_table_entry = None
    if use_blending_reference_table_entry:
      blending_reference_table_entry = blending_reference_table[blend_node]
      # print('using entry')
      # pprint(blending_reference_table_entry)

      # ─── EDIT 2 ───
      # **ensure** no input has zero QoA before we start dividing
      for node_idx, inp_metrics in blending_reference_table_entry.inputs.items():
          if inp_metrics.quality_of_answer <= 0:
              inp_metrics.quality_of_answer = DEFAULT_QOA

    ## now the existing code
    ncost, nlatency, nenergy, qoa_estimation, delta = 0.0, 0.0, 0.0, 0.0, 0.0
    output_tokens = single_model_metrics[ llm_assignments[blend_node] ] \
                        .average_output_tokens
    input_tokens  = query_tokens + blending_prompt_tokens

    # print('printing all the processing table entires')
    # pprint(processing_table_entries)

    for entry in processing_table_entries:
        # add inputs avg tokens
        input_tokens += entry.metrics.average_output_tokens

        # accumulate cost & energy, track max latency
        ncost    += entry.accumulated_metrics.final_cost
        nlatency  = max(nlatency, entry.accumulated_metrics.final_latency)
        nenergy  += entry.accumulated_metrics.final_energy

        # delta calculations for QoA
        final   = entry.accumulated_metrics.quality_of_answer

        if use_blending_reference_table_entry:
          # ─── EDIT 3 ───
          # use .get(...) and fallback again just in case
          initial = blending_reference_table_entry.inputs.get(entry.node) \
                        .quality_of_answer
          # now guaranteed >0
          # print(f'final: {final} inital: {initial} adding to delta {(final - initial) / initial}')
          delta += (final - initial) / initial

    # print('input tokens', input_tokens)
    # print('output tokens', output_tokens)
    # print('input cost', blending_node_metrics.input_cost)
    # print('output cost', blending_node_metrics.output_cost)
    # print('input latency', blending_node_metrics.input_latency)
    # print('output latency', blending_node_metrics.output_latency)
    # print('input energy', blending_node_metrics.input_energy)
    # print('output energy', blending_node_metrics.output_energy)


    
    token_cost    = blending_node_metrics.input_cost    * input_tokens \
                  + blending_node_metrics.output_cost   * output_tokens
    token_latency = blending_node_metrics.input_latency * input_tokens \
                  + blending_node_metrics.output_latency * output_tokens
    token_energy  = blending_node_metrics.input_energy  * input_tokens \
                  + blending_node_metrics.output_energy  * output_tokens

    # print('nlatency', nlatency)
    # print('token_late', token_latency)


    ncost    += token_cost
    nlatency += token_latency
    nenergy  += token_energy

    if use_blending_reference_table_entry:
      current_metrics.quality_of_answer = blending_reference_table_entry.output * (1 + delta / len(processing_table_entries))
    else:
      current_metrics.quality_of_answer = DEFAULT_QOA

    current_metrics.final_cost       = ncost
    current_metrics.final_latency    = nlatency
    current_metrics.final_energy     = nenergy
    return current_metrics





# NEW
def new_traversal_v3(
    G: nx.DiGraph,
    pairwise_metrics, # v6
    single_model_metrics, # v6
    matched_nodes_information, # v6
    blending_node_db_references,
    query_tokens:int,
    ctx_prompt_tokens:int,
    blending_prompt_tokens:int,
    whole_graph_llm_assignment: List[str]
):
  # print('in traversal')
  # plot_graph(G)
  import collections
  queue = collections.deque()
  processing_table = collections.defaultdict(list)
  processed_sinks = set()
  single_model_metrics = {
  int(k): v for k, v in single_model_metrics.items()
  }
  blending_node_db_references = {
    int(k): v for k, v in blending_node_db_references.items()
  }
  pairwise_metrics = {
    (int(a), int(b)): v
    for (a, b), v in pairwise_metrics.items()
  }

  # print('single_model_metrics')
  # pprint(single_model_metrics)

# ─── EDIT B: coerce assignment list entries to int ───
  whole_graph_llm_assignment = [int(x) for x in whole_graph_llm_assignment]

  llm_assignments = {node: whole_graph_llm_assignment[node] for node in G.nodes()}
  final_metrics = None

  ### EDGE CASE: only one node in the graph
  start_nodes = list(filter(lambda node: G.in_degree(node) == 0, G.nodes()))
  # print(f'start_nodes: {start_nodes}')
  # print(f'blending_reference {blending_node_db_references}')
  # print('start_nodes', start_nodes)
  # print('llm assignment')
  # pprint(llm_assignments)
  if len(start_nodes) == 1 and G.out_degree(start_nodes[0]) == 0:
    metrics = single_model_metrics[llm_assignments[start_node]]
    return LLMMetrics(
        final_cost=metrics.final_cost,
        final_latency=metrics.final_latency,
        final_energy=metrics.final_energy,
        quality_of_answer=metrics.quality_of_answer
    )

  # print('s', start_nodes)

  for start_node in start_nodes:
    key = llm_assignments[start_node]
    metrics = copy.deepcopy(single_model_metrics[key])
    # zero out the final cost and final latency because we are calculating token-wise.
    # and the start nodes are still single models, so they won't be matched as a sub-schedule
    metrics.final_cost = 0
    metrics.final_latency = 0
    metrics.final_energy = 0
    queue.append(BFSNode(
        LLM_name=key,
        node=start_node,
        metrics=metrics,
        accumulated_metrics=metrics,
        first_node_in_seq=True
    ))

  # print('initial state of the queue:')
  # pprint(queue)
  while queue:
    bfs_node = queue.popleft()
    LLM_name, node, metrics, current_metrics, first_node_in_seq = bfs_node

    # metrics = copy.deepcopy(metrics)
    # current_metrics = copy.deepcopy(current_metrics)
    # print('processing from queue', LLM_name)

    # might not need second part of AND condition
    if is_blend_node(G, node) and node not in matched_nodes_information:
      # Node has not received information from all its inputs, so reprocess at later time
      if len(processing_table[node]) != G.in_degree(node):
        queue.append(bfs_node)
        continue
      else:
        # calc_cost_parallel will return new reference of current_metrics, so original is not modified
        # print('p table for', node_key, node)
        # pprint(processing_table[node])
        current_metrics = calculate_cost_parallel(
            llm_assignments,
            node,
            current_metrics,
            metrics,
            processing_table[node],
            blending_node_db_references,
            single_model_metrics,
            query_tokens,
            blending_prompt_tokens
        )

    successors = list(G.successors(node))
    if len(successors) == 0 and len(queue) == 0:
      final_metrics = current_metrics
      break

    # print(f'successors for {successors} for {llm_assignments[node]}')
    for successor in successors:
      # these keys are the LLM names based off their corresponding node integer
      node_key, successor_key = llm_assignments[node], llm_assignments[successor]
      # we came across a successor that is apart of a matched sub-schedule, but because it is in processed_sink, that means this successor has
      # already been added to the queue, hence seen, so we do not do anything here
      if successor in matched_nodes_information and successor in processed_sinks:
        continue

      # matched sub-schedule sink found, so we merely propagate and do not touch the associated metrics
      if successor in matched_nodes_information:
        processed_sinks.add(successor)
        metrics = matched_nodes_information[successor][0] # the 0 index is here because matched_node_information[key] points to list of info, 0th element has the metrics
        if isinstance(metrics, tuple):
          metrics = convert_metrics_to_dict(metrics)
        queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key], accumulated_metrics=metrics, first_node_in_seq=False))
        continue

      if is_blend_node(G, successor):
        # print(f'in here, current_node: {node_key}, successor: {successor_key} type node: {type(node_key)} type succ: {type(successor_key)}')
        # if we encounter a blend node, but the current node has no preds that means it is a single model
        # so we set the final_cost, latency, and energy to 0, because

        ncurrent_metrics = copy.deepcopy(current_metrics)
        # we do not use the calculate_cost_sequential_v2 here because the behavior of calculation is slightly different
        if is_start_node(G, node):
          # print('in here for LLM', LLM_name)
          cost = query_tokens * metrics.input_cost + metrics.average_output_tokens * metrics.output_cost
          latency = query_tokens * metrics.input_latency + metrics.average_output_tokens * metrics.output_latency
          energy = query_tokens * metrics.input_energy + metrics.average_output_tokens * metrics.output_energy
          # print(metrics.input_cost, metrics.input_energy)
          # print('node is first and succ is blend', cost, latency, energy)
          ncurrent_metrics.final_cost = cost
          ncurrent_metrics.final_latency = latency
          ncurrent_metrics.final_energy = energy

        # if node is sequential node, and there are predecessors, then we assume that the cost incurred
        # by this node has been calculated previously when this node was a successor for something else
        # Also, if node is blend node, we can just propagate, because metric acclumation would have already happened in the top of the while loop
        processing_table[successor].append(ProcessingTableEntry(node=node, LLM=node_key, metrics=metrics, accumulated_metrics=ncurrent_metrics))
        # print('current state of processing table', processing_table[successor])
        # first time reaching blend node, so we will add it to the queue for processing, remember at the top of the while loop, we will readd the same blend node if
        # we do not see the required amount of information in the processing table
        if successor not in processed_sinks:
          processed_sinks.add(successor)
          # if successor_key != 5 :
          queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key], accumulated_metrics=single_model_metrics[successor_key], first_node_in_seq=False))

      # Case where successor is sequential, we handle metric aggregation different based on if it is the first node in the sequential chain
      else:
        # Case where it is the first node. In this case, we will add the costs incurred by A and B, also we know that node is not a blend node
        if first_node_in_seq:
          # print('accessing pairwise metrics for ', (node_key, successor_key))
          if node_key != 5 or successor_key != 5:
            ncurrent_metrics = copy.deepcopy(current_metrics)
            pairwise_execution_metrics = pairwise_metrics[(node_key, successor_key)]
            # these are current metrics for single model, so we can replace final_cost, final_latency, final_energy because we are calculating token wise
            ncurrent_metrics.quality_of_answer = pairwise_execution_metrics.quality_of_answer
            # print('doing cost_seq for ', node_key, successor_key)
            cost, latency, energy = calculate_cost_sequential_v2(
                query_tokens,
                metrics.average_output_tokens,
                ctx_prompt_tokens,
                single_model_metrics[successor_key].average_output_tokens,
                metrics,
                single_model_metrics[successor_key],
                True
            )
            ncurrent_metrics.final_cost = cost
            ncurrent_metrics.final_latency = latency
            ncurrent_metrics.final_energy = energy
            # print('finished seq cost first node in seq')
            # print(ncurrent_metrics.final_cost, ncurrent_metrics.final_latency, ncurrent_metrics.final_energy)
            queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key], accumulated_metrics=ncurrent_metrics, first_node_in_seq=False))
        else:
          successor_metrics = single_model_metrics[successor_key]
          ncurrent_metrics = copy.deepcopy(current_metrics)
          if is_blend_node(G, node):
            # qoa_estimation = (current_metrics.quality_of_answer + successor_metrics.quality_of_answer) / 2
            # node -> successor
            # def sequential_delta(node_A: str, node_B:str, single_model_metrics, pairwise_metrics):
            #   final = pairwise_metrics[(node_A, node_B)].quality_of_answer
            #   initial = single_model_metrics[node_A].quality_of_answer
            #   delta = (final - initial) - initial
            #   return delta
            N = 0
            delta = 0
            for llm in llm_assignments:
              if G.in_degree(llm) >= 2:
                continue
              N += 1
              delta += sequential_delta(llm_assignments[llm], llm_assignments[successor], single_model_metrics, pairwise_metrics)
            qoa_estimation = ncurrent_metrics.quality_of_answer * (1 + (delta / N))
          else:
            qoa_estimation = ncurrent_metrics.quality_of_answer * (1 + sequential_delta(llm_assignments[node], llm_assignments[successor], single_model_metrics, pairwise_metrics))

          cost, latency, energy = calculate_cost_sequential_v2(
              query_tokens,
              single_model_metrics[node_key].average_output_tokens,
              ctx_prompt_tokens,
              successor_metrics.average_output_tokens,
              None,
              successor_metrics,
              False,
          )
          # add to queue after aggregation
          ncurrent_metrics.final_cost += cost
          ncurrent_metrics.final_latency += latency
          ncurrent_metrics.final_energy += energy
          ncurrent_metrics.quality_of_answer = qoa_estimation
          queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key], accumulated_metrics=ncurrent_metrics, first_node_in_seq=False))

      # print('sadasd')
      # pprint(queue)

    # print('state of queue')
    # pprint(queue)

  return final_metrics


def new_traversal_just_for_cost_latency_energy_v3(
    G: nx.DiGraph,
    pairwise_metrics, # v6
    single_model_metrics, # v6
    matched_nodes_information, # v6
    blending_node_db_references,
    query_tokens:int,
    ctx_prompt_tokens:int,
    blending_prompt_tokens:int,
    whole_graph_llm_assignment: List[str]
):
  # print('in new_traversal_just_for_cost_latency_energy_v3')
  # plot_graph(G)
  import collections
  queue = collections.deque()
  processing_table = collections.defaultdict(list)
  processed_sinks = set()
  single_model_metrics = {
  int(k): v for k, v in single_model_metrics.items()
  }
  blending_node_db_references = {
    int(k): v for k, v in blending_node_db_references.items()
  }
  pairwise_metrics = {
    (int(a), int(b)): v
    for (a, b), v in pairwise_metrics.items()
  }

  # print('single_model_metrics')
  # pprint(single_model_metrics)

# ─── EDIT B: coerce assignment list entries to int ───
  whole_graph_llm_assignment = [int(x) for x in whole_graph_llm_assignment]

  llm_assignments = {node: whole_graph_llm_assignment[node] for node in G.nodes()}
  final_metrics = None

  ### EDGE CASE: only one node in the graph
  start_nodes = list(filter(lambda node: G.in_degree(node) == 0, G.nodes()))
  # print(f'start_nodes: {start_nodes}')
  # print(f'blending_reference {blending_node_db_references}')
  # print('start_nodes', start_nodes)
  # print('llm assignment')
  # pprint(llm_assignments)
  if len(start_nodes) == 1 and G.out_degree(start_nodes[0]) == 0:
    metrics = single_model_metrics[llm_assignments[start_node]]
    return LLMMetrics(
        final_cost=metrics.final_cost,
        final_latency=metrics.final_latency,
        final_energy=metrics.final_energy,
        quality_of_answer=metrics.quality_of_answer
    )

  # print('s', start_nodes)

  for start_node in start_nodes:
    key = llm_assignments[start_node]
    metrics = copy.deepcopy(single_model_metrics[key])
    # zero out the final cost and final latency because we are calculating token-wise.
    # and the start nodes are still single models, so they won't be matched as a sub-schedule
    metrics.final_cost = 0
    metrics.final_latency = 0
    metrics.final_energy = 0
    queue.append(BFSNode(
        LLM_name=key,
        node=start_node,
        metrics=metrics,
        accumulated_metrics=metrics,
        first_node_in_seq=True,
    ))

  # print('initial state of the queue:')
  # pprint(queue)
  while queue:
    bfs_node = queue.popleft()
    LLM_name, node, metrics, current_metrics, first_node_in_seq = bfs_node

    # metrics = copy.deepcopy(metrics)
    # current_metrics = copy.deepcopy(current_metrics)
    # print('processing from queue', LLM_name)

    # might not need second part of AND condition
    if is_blend_node(G, node) and node not in matched_nodes_information:
      # Node has not received information from all its inputs, so reprocess at later time
      if len(processing_table[node]) != G.in_degree(node):
        queue.append(bfs_node)
        continue
      else:
        # calc_cost_parallel will return new reference of current_metrics, so original is not modified
        # print('p table for', node_key, node)
        # pprint(processing_table[node])
        current_metrics = calculate_cost_parallel(
            llm_assignments,
            node,
            current_metrics,
            metrics,
            processing_table[node],
            blending_node_db_references,
            single_model_metrics,
            query_tokens,
            blending_prompt_tokens,
            turn_off_blend_table_check=True
        )

    successors = list(G.successors(node))
    if len(successors) == 0 and len(queue) == 0:
      final_metrics = current_metrics
      break

    # print(f'successors for {successors} for {llm_assignments[node]}')
    for successor in successors:
      # these keys are the LLM names based off their corresponding node integer
      node_key, successor_key = llm_assignments[node], llm_assignments[successor]
      # we came across a successor that is apart of a matched sub-schedule, but because it is in processed_sink, that means this successor has
      # already been added to the queue, hence seen, so we do not do anything here
      if successor in matched_nodes_information and successor in processed_sinks:
        continue

      # matched sub-schedule sink found, so we merely propagate and do not touch the associated metrics
      if successor in matched_nodes_information:
        processed_sinks.add(successor)
        metrics = matched_nodes_information[successor][0] # the 0 index is here because matched_node_information[key] points to list of info, 0th element has the metrics
        if isinstance(metrics, tuple):
          metrics = convert_metrics_to_dict(metrics)
        #queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=metrics, accumulated_metrics=metrics, first_node_in_seq=False))
        queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key], accumulated_metrics=metrics, first_node_in_seq=False))
        continue

      if is_blend_node(G, successor):
        # print(f'in here, current_node: {node_key}, successor: {successor_key} type node: {type(node_key)} type succ: {type(successor_key)}')
        # if we encounter a blend node, but the current node has no preds that means it is a single model
        # so we set the final_cost, latency, and energy to 0, because

        ncurrent_metrics = copy.deepcopy(current_metrics)
        # we do not use the calculate_cost_sequential_v2 here because the behavior of calculation is slightly different
        if is_start_node(G, node):
          # print('in here for LLM', LLM_name)
          cost = query_tokens * metrics.input_cost + metrics.average_output_tokens * metrics.output_cost
          latency = query_tokens * metrics.input_latency + metrics.average_output_tokens * metrics.output_latency
          energy = query_tokens * metrics.input_energy + metrics.average_output_tokens * metrics.output_energy
          # print(metrics.input_cost, metrics.input_energy)
          # print('node is first and succ is blend', cost, latency, energy)
          ncurrent_metrics.final_cost = cost
          ncurrent_metrics.final_latency = latency
          ncurrent_metrics.final_energy = energy

        # if node is sequential node, and there are predecessors, then we assume that the cost incurred
        # by this node has been calculated previously when this node was a successor for something else
        # Also, if node is blend node, we can just propagate, because metric acclumation would have already happened in the top of the while loop
        processing_table[successor].append(ProcessingTableEntry(node=node, LLM=node_key, metrics=metrics, accumulated_metrics=ncurrent_metrics))
        # print('current state of processing table', processing_table[successor])
        # first time reaching blend node, so we will add it to the queue for processing, remember at the top of the while loop, we will readd the same blend node if
        # we do not see the required amount of information in the processing table
        if successor not in processed_sinks:
          processed_sinks.add(successor)
          # if successor_key != 5 :
          queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key], accumulated_metrics=single_model_metrics[successor_key], first_node_in_seq=False))

      # Case where successor is sequential, we handle metric aggregation different based on if it is the first node in the sequential chain
      else:
        # Case where it is the first node. In this case, we will add the costs incurred by A and B, also we know that node is not a blend node
        if first_node_in_seq:
          # print('accessing pairwise metrics for ', (node_key, successor_key))
          if node_key != 5 or successor_key != 5:
            ncurrent_metrics = copy.deepcopy(current_metrics)
            # pairwise_execution_metrics = pairwise_metrics[(node_key, successor_key)]
            # these are current metrics for single model, so we can replace final_cost, final_latency, final_energy because we are calculating token wise
            # ncurrent_metrics.quality_of_answer = pairwise_execution_metrics.quality_of_answer
            # print('doing cost_seq for ', node_key, successor_key)
            cost, latency, energy = calculate_cost_sequential_v2(
                query_tokens,
                metrics.average_output_tokens,
                ctx_prompt_tokens,
                single_model_metrics[successor_key].average_output_tokens,
                metrics,
                single_model_metrics[successor_key],
                True
            )
            ncurrent_metrics.final_cost = cost
            ncurrent_metrics.final_latency = latency
            ncurrent_metrics.final_energy = energy
            # print('finished seq cost first node in seq')
            # print(ncurrent_metrics.final_cost, ncurrent_metrics.final_latency, ncurrent_metrics.final_energy)
            queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key], accumulated_metrics=ncurrent_metrics, first_node_in_seq=False))
        else:
          successor_metrics = single_model_metrics[successor_key]
          ncurrent_metrics = copy.deepcopy(current_metrics)
          # if is_blend_node(G, node):
          #   # qoa_estimation = (current_metrics.quality_of_answer + successor_metrics.quality_of_answer) / 2
          #   # node -> successor
          #   # def sequential_delta(node_A: str, node_B:str, single_model_metrics, pairwise_metrics):
          #   #   final = pairwise_metrics[(node_A, node_B)].quality_of_answer
          #   #   initial = single_model_metrics[node_A].quality_of_answer
          #   #   delta = (final - initial) - initial
          #   #   return delta
          #   N = 0
          #   delta = 0
          #   for llm in llm_assignments:
          #     if G.in_degree(llm) >= 2:
          #       continue
          #     N += 1
          #     delta += sequential_delta(llm_assignments[llm], llm_assignments[successor], single_model_metrics, pairwise_metrics)
          #   qoa_estimation = ncurrent_metrics.quality_of_answer * (1 + (delta / N))
          # else:
          #   qoa_estimation = ncurrent_metrics.quality_of_answer * (1 + sequential_delta(llm_assignments[node], llm_assignments[successor], single_model_metrics, pairwise_metrics))

          cost, latency, energy = calculate_cost_sequential_v2(
              query_tokens,
              single_model_metrics[node_key].average_output_tokens,
              ctx_prompt_tokens,
              successor_metrics.average_output_tokens,
              None,
              successor_metrics,
              False,
          )
          # add to queue after aggregation
          ncurrent_metrics.final_cost += cost
          ncurrent_metrics.final_latency += latency
          ncurrent_metrics.final_energy += energy
          # ncurrent_metrics.quality_of_answer = qoa_estimation
          queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key], accumulated_metrics=ncurrent_metrics, first_node_in_seq=False))

      # print('sadasd')
      # pprint(queue)

    # print('state of queue')
    # pprint(queue)

  return final_metrics

def get_sub_llm_assignment(subDAG, llm_assignment):
  return [llm_assignment[node] for node in sorted(subDAG.nodes())]

def get_seq_chains_starting_from(g, node, chains, path=None):
    # print('current path', path, 'current node', node, 'successor for this node', g.successors(node))
    if is_blend_node(g, node):
        chains.append(path)
        return

    if path is None:
        path = []

    if g.out_degree(node) == 0:
        chains.append(path + [node])
        return

    path = path + [node]  # create a new path to avoid mutation
    for successor in g.successors(node):
        get_seq_chains_starting_from(g, successor, chains, path)

def flatten_intermediate_blends_stagewise(G):
  """
  Remove only blending nodes (indegree>1) that feed into other blending nodes.
  For each such node, reattach its predecessors directly to its successors, then remove it.
  Repeat until no blend->blend edges remain.
  """
  H = G.copy()
  while True:
    # find blend nodes feeding other blend nodes
    blend_nodes = [
        n for n in H.nodes()
        if H.in_degree(n) > 1 and any(H.in_degree(s) > 1 for s in H.successors(n))
    ]
    if not blend_nodes:
      break
    n = blend_nodes[0]
    preds = list(H.predecessors(n))
    succs = [s for s in H.successors(n) if H.in_degree(s) > 1]
    # reattach preds -> those blend successors
    for p in preds:
      for s in succs:
        if p != s and not H.has_edge(p, s):
          H.add_edge(p, s)
    # remove n's blend->blend connections and node
    H.remove_node(n)
  return H

def flatten_intermediate_blends_stagewise_mut(H):
  """
  Remove only blending nodes (indegree>1) that feed into other blending nodes.
  For each such node, reattach its predecessors directly to its successors, then remove it.
  Repeat until no blend -> blend edges remain.
  """
  while True:
    # find blend nodes feeding other blend nodes
    blend_nodes = [
        n for n in H.nodes()
        if H.in_degree(n) > 1 and any(H.in_degree(s) > 1 for s in H.successors(n))
    ]
    if not blend_nodes:
      break
    n = blend_nodes[0]
    preds = list(H.predecessors(n))
    succs = [s for s in H.successors(n) if H.in_degree(s) > 1]
    # reattach preds -> those blend successors
    for p in preds:
      for s in succs:
        if p != s and not H.has_edge(p, s):
          H.add_edge(p, s)
    # remove n's blend->blend connections and node
    H.remove_node(n)

## NEW
"""
V3 matches starting sequential chains from the back of the chain
- after sequential chain matching, we fetch reference models for the blending nodes for parallel estimation
- using matches and the parallel strategy, we estimate any unmatched sequential components with the pairwise method
"""
def match_subDAGs_v3(
  G: nx.DiGraph,
  assignment: List[str],
  query_type: str,
  df_history: pd.DataFrame,
  single_model_metrics,
  pairwise_model_metrics,
  levenshtein_threshold: float=0.75,
  turn_off_exact_fuzzy_matching=False,
  print_debug_log=True
):
  import collections
  debug_log = []
  matched_nodes_information = collections.defaultdict(list)
  # Get the final LLM/node in DAG to start our DFS from
  sink = list(filter(lambda node: G.out_degree(node) == 0, G.nodes()))[0]
  # construct the DAG behind sink
  subDAG, sub_llm_assignment, sink = construct_subdag_behind_node(G, sink, assignment)
  target_structure_id = canonical_label_for_isomorphism(subDAG)

  # First we fuzzy match the entire graph
  # check for exact match OR a very good fuzzy match

  # print('whole Graph', assignment)
  # plot_graph(G)
  if not turn_off_exact_fuzzy_matching:
    last_node_only = nx.DiGraph()
    last_node_only.add_node(sink)
    sink_node_llm = get_sub_llm_assignment(last_node_only, assignment)[0]
    best_candidate_llm_assignment, metrics, score = fuzzy_ranked_matches_from_database(
      sub_llm_assignment,
      target_structure_id,
      sink_node_llm,
      query_type,
      df_history,
      threshold=levenshtein_threshold,
    )

    # suitable match is found so let's store the metrics in our dictionary
    if metrics is not None and score >= levenshtein_threshold:
      # _metrics = (cost, latency, energy, qoa) : FORMAT
      cost, latency, energy, qoa = metrics
      matched_nodes_information[sink].append(LLMMetrics(final_cost=cost, final_latency=latency, final_energy=energy, quality_of_answer=qoa))
      matched_nodes_information[sink].append('fuzzy-match')
      matched_nodes_information[sink].append(best_candidate_llm_assignment)
      matched_nodes_information[sink].append(score)
      return metrics, {}, False # False here tells the estimation function no need for traversal because we found a good exact OR fuzzy match on whole graph


    start_nodes = list(filter(lambda node: G.in_degree(node) == 0, G.nodes()))
    # print('start_nodes', start_nodes)
    sequential_chains = []
    for node in start_nodes:
      get_seq_chains_starting_from(G, node, sequential_chains)

    # the sequential chains [[[0, 1]], [[2, 3, 4]], [[6]]]
    # print(f'Sequential chains: {sequential_chains}')
    for seq_chain in sequential_chains:
      # ignore single models
      if len(seq_chain) == 1:
        continue
      # Temp Graph that we can afford to mutate structure
      TG = nx.DiGraph()
      TG.add_nodes_from(seq_chain)
      for i in range(1, len(seq_chain)):
        TG.add_edge(seq_chain[i - 1], seq_chain[i])

      # issue here, when using different start node in chain function, the ordering don't match up
      TG_llm_assignment = get_sub_llm_assignment(TG, assignment)
      # print('Graph of seq chain going into fuzzy', seq_chain, TG_llm_assignment)
      # plot_graph(TG)
      metrics, matchedG, last_node_in_sequential = fuzzy_match_sequential_chain(TG, TG_llm_assignment, query_type, single_model_metrics, df_history, levenshtein_threshold)
      # print('last node in sequential', last_node_in_sequential)
      if metrics is not None:
        matched_nodes_information[last_node_in_sequential].append(metrics)
        matched_nodes_information[last_node_in_sequential].append('sequential-match')

  # pprint(matched_nodes_information)
  # print('jj')
  # at this point fuzzy matching the entire graph is done, so we now merge the blend node dependencies
  flatten_intermediate_blends_stagewise_mut(G)
  # new llm assignment after removing the blend nodes
  # assignment = get_sub_llm_assignment_given_nodes(G.nodes(), assignment)
  # plot_graph(G)
  # print('llm assignment after flatten func', assignment)

  # here we go through each blend node, and gather its inputs, because of the flatten function call
  # if a blend nodes input was a blend node, the original blend node gets that blend nodes input
  # we look for matches on the blend_node and its input in the DB and store the qoa along with the qoa
  # of the input models
  blend_nodes = filter(lambda node: G.in_degree(node) >=2, list(nx.topological_sort(G)))
  # print('blend_nodes', blend_nodes)
  # print('plotting graph after flattening jizzz')
  # plot_graph(G)
  # print('graph plotted')
  blending_node_db_references = {}
  for blend_node in blend_nodes:
    # print('in the blend loop')

    # create the dummy dag for searching
    inputs = list(G.predecessors(blend_node))
    llm_to_single_model_qoa_for_reference = {}
    dummy_DAG = nx.DiGraph()
    dummy_DAG.add_nodes_from(inputs + [blend_node])

    # print('nodes added to dummy dag', list(dummy_DAG.nodes()))

    for input in inputs:
      dummy_DAG.add_edge(input, blend_node)
      the_llm = get_sub_llm_assignment_given_nodes([input], assignment)[0]
      llm_to_single_model_qoa_for_reference[input] = single_model_metrics[the_llm]

    # get the llm_assignment for searching
    dummy_DAG_llm_assignment = get_sub_llm_assignment(dummy_DAG, assignment)
    # print('dummy lm', dummy_DAG_llm_assignment)
    # print('dummy_DAG_llm_assignment', dummy_DAG_llm_assignment)
    # plot_graph(dummy_DAG)

    # search
    # bn_reference_metrics = get_subdag_metrics_v7(
    #   dummy_DAG,
    #   dummy_DAG_llm_assignment,
    #   query_type,
    #   df_history
    # )

    bn_reference_metrics = special_get_subdag_metrics_for_one_blend_operations(
      dummy_DAG,
      dummy_DAG_llm_assignment,
      query_type,
      df_history
    )


    # based on our level scheme we are guaranteed to have them
    # print('blend graph')
    # plot_graph(dummy_DAG)
    # print('dummy dag nodes', list(dummy_DAG.nodes()))
    # for edge in dummy_DAG.edges:
    #   print('edge', edge)
    # print('blend graph llm assignment', dummy_DAG_llm_assignment)
    # print('original graph llm assignment', assignment)

    # UNIQUE COMMENT FOR BHARG, error happens because there is no match in DB for our parallel subgraph
    if bn_reference_metrics:
      blending_node_db_references[blend_node] = BlendingNodeDBReference(
          output=bn_reference_metrics.quality_of_answer,
          inputs=llm_to_single_model_qoa_for_reference
      )
    else:
      print("no bn_ref metrics found for", blend_node)

  return matched_nodes_information, blending_node_db_references, True # True here indicates we need to perform the traversal for stats

def estimate_schedule_v3(
    G: nx.DiGraph,
    assignment: List[str],
    query_type: str,
    query_tokens: int,
    blending_prompt_tokens: int,
    ctx_tokens:int,
    df_history: pd.DataFrame,
    levenshtein_threshold: float=0.75,
    turn_off_exact_fuzzy_matching=False
):
  # this function will return the last blend node and its inputs, if graph is full sequential then the returned dictionary will only contain content if there is a good fuzzy/exact match for
  # the full sequential graph
  original_graph_copy = G.copy()
  single_model_metrics = get_single_model_metrics(assignment, query_type, df_history)
  pairwise_metrics = get_pairwise_metrics(assignment, query_type, df_history)

  if len(single_model_metrics) == 0:
    # level zero return default stats
    return LLMMetrics(
      final_cost=0.5,
      final_latency=20,
      final_energy=20,
      quality_of_answer=0.5
    )
  elif len(pairwise_metrics) == 0:
    # level 1, so we only have parallel and single model executions
    H = flatten_intermediate_blends_stagewise(G)
    sink = list(filter(lambda node: G.out_degree(node) == 0, G.nodes()))[0]
    # if last node is non-blending, based on information in the table, then most we can do is take the single-model stats
    if is_sequential_node(H, sink):
      # print('here 1')
      # print("here 1", single_model_metrics[assignment[sink]])
      return single_model_metrics[assignment[sink]]
    else:
      # if last node is blending, we will take that node and get its LLM inputs, and base our prediction off of that
      # because level will for sure have this execution in the table
      TG = nx.DiGraph()
      preds = list(H.predecessors(sink))
      TG.add_nodes_from([sink])
      for pred in preds:
        TG.add_edge(pred, sink)
      TG_llm_assignment = get_sub_llm_assignment(TG, assignment)
      # print('here 2', TG_llm_assignment)
      # plot_graph(TG)
      # print("here 2", get_subdag_metrics_v7(TG, TG_llm_assignment, query_type, df_history))
      #return get_subdag_metrics_v7(TG, TG_llm_assignment, query_type, df_history)
      return special_get_subdag_metrics_for_one_blend_operations(TG, TG_llm_assignment, query_type, df_history)


  # level 2 and above
  # first this function will fuzzy/exact match the entire graph, if cant be done, then fuzzy/exact match sequential chains from the start
  # then it gathers information on the blending node and their input's QoA, storing them in blending_node_db_references
  matched_subDAG_information, blending_node_db_references, need_traversal = match_subDAGs_v3(G, assignment, query_type, df_history, single_model_metrics, pairwise_metrics, levenshtein_threshold, turn_off_exact_fuzzy_matching=turn_off_exact_fuzzy_matching)
  # pprint(matched_subDAG_information)
  # pprint(matched_subDAG_information)
  if not need_traversal:
    # in this case matched_subDAG_information is actually a tuple if need_traversal is False
    cost, latency, energy, qoa = matched_subDAG_information
    # info = list(matched_subDAG_information.keys())[0]
    # cost = info["final_cost"]
    # energy = info["final_energy"]
    # latency = info["final_latency"]
    # qoa = info["quality_of_answer"]
    # print("here 3", cost, latency, energy, qoa)
    return LLMMetrics(
      final_cost=cost,
      final_latency=latency,
      final_energy=energy,
      quality_of_answer=qoa
    )

  # print('made here, starting traversal')
  # plot_graph(G)
  # print('og', assignment)
  # traverse the graph, with information on matches and blending nodes for qoa estimation
  # plot_graph(G)
  # print(assignment)
  # print('nodes', list(G.nodes()))
  # for edge in G.edges:
  #   print('edge', edge)
  # print('assignment before going to traversal', assignment)
  # plot_graph(G)
  # print('matched_subdag_info')
  # pprint(matched_subDAG_information)
  final_metrics = new_traversal_v3(
    G,
    pairwise_metrics,
    single_model_metrics,
    matched_subDAG_information,
    blending_node_db_references,
    query_tokens,
    ctx_tokens,
    blending_prompt_tokens,
    assignment,
  )
  # ('returning final metrics after traversal')
  if final_metrics is None:raise RuntimeError("new_traversal_v3 returned None: traversal never reached a terminal sink")

  final_qoa = final_metrics.quality_of_answer

  #print('blending_node_db_ref before new_trav')
  # pprint(blending_node_db_references)

  final_metrics_for_just_cost_energy_latency = new_traversal_just_for_cost_latency_energy_v3(
    original_graph_copy,
    pairwise_metrics,
    single_model_metrics,
    matched_subDAG_information,
    blending_node_db_references,
    query_tokens,
    ctx_tokens,
    blending_prompt_tokens,
    assignment,
  )
  final_metrics.final_cost = final_metrics_for_just_cost_energy_latency.final_cost
  final_metrics.final_energy = final_metrics_for_just_cost_energy_latency.final_energy
  final_metrics.final_latency = final_metrics_for_just_cost_energy_latency.final_latency
  #print('all the shit for schedule is done')
  return final_metrics

class Individual:
    def __init__(self, struct_id, assignment):
        self.struct_id = struct_id          # DAG structure identifier (canonical mask)
        self.assignment = tuple(assignment) # tuple of LLM model indices per node
        self.metrics = None                # (cost, latency, energy, qoa)
        self.objectives = None             # (cost, latency, energy, -qoa)
        self.rank = None
        self.crowding_distance = None

# Helper to get indegree list from structure
def get_indegrees(k, struct_id):
    indeg = [0]*k
    bit_index = 0
    for i in range(k-1):
        for j in range(i+1, k):
            # print("Left shift for", i, ",", j, "is", (1 << bit_index))
            if struct_id & (1 << bit_index):
                indeg[j] += 1
            bit_index += 1
    return indeg

# Initialize population with random DAGs and valid assignments
def initialize_population(pop_size, max_nodes):
    structures = generate_nonisomorphic_dags(max_nodes)
    population = []
    for _ in range(pop_size):
        k, struct_id = random.choice(structures)
        indeg = get_indegrees(k, struct_id)
        # Randomly assign base models, set blending for indegree>1
        assignment = [None]*k
        blending_models = 0
        for i in range(k):
            assignment[i] = 5 if indeg[i] > 1 else random.randint(0, 4)
            if assignment[i] == 5: blending_models += 1
        # print('ip', assignment, blending_models)

        # bandage solution added:
        # is_valid = True
        # # if blending_models > 1 or blending_models == 1 and assignment[-1] != 5:
        # #   is_valid = False
        # if blending_models >= 1 and assignment[-1] != 5:
        #    is_valid = False

        # if is_valid:
        individual = Individual(struct_id, assignment)
        population.append(individual)

    return population

# Crossover operator
def crossover(parent1, parent2):
    # If structures have different sizes (node counts), skip crossover (just clone one parent)
    if len(parent1.assignment) != len(parent2.assignment):
        return copy.deepcopy(parent1 if random.random() < 0.5 else parent2)
    k = len(parent1.assignment)
    # One-point crossover on the edge bitstring
    num_bits = k*(k-1)//2
    mask1, mask2 = parent1.struct_id, parent2.struct_id
    if num_bits > 1:
        cp = random.randint(1, num_bits-1)
        # Combine bits up to cp from parent1, rest from parent2
        new_mask = ((mask1 & ((1<<cp)-1)) | (mask2 & ~((1<<cp)-1)))
    else:
        # new_mask = 0
        # too few bits to crossover meaningfully: just pick one parent’s mask
        new_mask = mask1 if random.random() < 0.5 else mask2
    # One-point crossover on assignment
    assign1, assign2 = list(parent1.assignment), list(parent2.assignment)
    if k > 1:
        cp2 = random.randint(1, k-1)
        new_assign = assign1[:cp2] + assign2[cp2:]
    else:
        new_assign = assign1[:]  # only one node
    # Repair the structure to enforce single sink connectivity
    # Ensure only last node has no outgoing: if any node < last has outdeg 0, connect it to last node
    bit_index = 0
    outdeg = [0]*k
    for i in range(k-1):
        for j in range(i+1, k):
            if new_mask & (1<<bit_index):
                outdeg[i] += 1
            bit_index += 1
    for node in range(k-1):
        if outdeg[node] == 0:  # no outgoing edge for a non-sink
            # Add edge from this node to sink (node -> k-1)
            # Compute bit index for (node, k-1)
            idx = 0
            for a in range(node):
                idx += (k-1 - a)
            idx += (k-1 - node - 1)
            new_mask |= (1 << idx)
    # Repair model assignments (assign blending model if indegree>1, otherwise ensure base)
    indeg = get_indegrees(k, new_mask)
    # print("indegrees", indeg)
    for i in range(k):
        if indeg[i] > 1:
            new_assign[i] = 5
        elif indeg[i] <= 1 and new_assign[i] == 5:
            new_assign[i] = random.randint(0, 4)
    # Canonicalize the offspring’s structure (find highest bit representation)
    # Reconstruct adjacency matrix to apply canonical_representation
    adj = [[0]*k for _ in range(k)]
    bit_index = 0
    for i in range(k-1):
        for j in range(i+1, k):
            if new_mask & (1 << bit_index):
                adj[i][j] = 1
            bit_index += 1
    canon_mask = canonical_representation(adj, k)
    indeg = get_indegrees(k, canon_mask)
    # print("indegrees", indeg)
    for i in range(k):
        if indeg[i] > 1:
            new_assign[i] = 5
        elif indeg[i] <= 1 and new_assign[i] == 5:
            new_assign[i] = random.randint(0, 4)
    # If canonical relabeling changed node indices, we should permute the assignment accordingly
    # We find the permutation that produces canon_mask from adj
    best_perm = None
    nodes = list(range(k))
    for perm in itertools.permutations(nodes):
        # apply perm as labeling and compute bitmask
        perm_adj = [[0]*k for _ in range(k)]
        valid = True
        for u in range(k):
            for v in adj[u]:
                pu, pv = perm.index(u), perm.index(v)
                if pu > pv:
                    valid = False
                    break
                perm_adj[pu][pv] = 1
            if not valid: break
        if not valid: continue
        mask = 0; idx = 0
        for a in range(k-1):
            for b in range(a+1, k):
                if perm_adj[a][b] == 1:
                    mask |= (1 << idx)
                idx += 1
        if mask == canon_mask:
            best_perm = perm
            break
    if best_perm:
        permuted_assign = [None]*k
        for old_node in range(k):
            new_idx = best_perm.index(old_node)
            permuted_assign[new_idx] = new_assign[old_node]
        new_assign = permuted_assign
    # Return new Individual
    return Individual(canon_mask, new_assign)

# Mutation operator
def mutate(ind, max_nodes):
    k = len(ind.assignment)
    struct_id = ind.struct_id
    assign = list(ind.assignment)
    # With small probability, add a new node (increase DAG size)
    # if k < max_nodes and random.random() < 0.1:
    if k < max_nodes and random.random() < ADD_NODE_PROB:
        new_k = k + 1
        # Create new adjacency matrix with new node as sink
        adj = [[0]*new_k for _ in range(new_k)]
        bit_index = 0
        for i in range(k-1):
            for j in range(i+1, k):
                if struct_id & (1 << bit_index):
                    adj[i][j] = 1
                bit_index += 1
        # Connect old sink (node k-1) to new node (node k)
        adj[k-1][k] = 1
        # New node assignment: since indegree will be 1, assign a random base model
        assign.append(random.randint(0, 4))
        # Compute canonical mask and adjust assignment order accordingly
        canon_mask = canonical_representation(adj, new_k)
        # (For simplicity, assume the labeling with new node as highest is canonical)
        return Individual(canon_mask, assign)
    # Otherwise, mutate existing structure or model assignments
    # Flip an edge with some probability
    # if k > 1 and random.random() < 0.3:
    if k > 1 and random.random() < FLIP_EDGE_PROB:
        num_bits = k*(k-1)//2
        bit_to_flip = random.randrange(num_bits)
        struct_id ^= (1 << bit_to_flip)  # toggle the chosen edge
    # Mutate model assignments: random base model change for some nodes
    for i in range(k):
        # if assign[i] != 5 and random.random() < 0.2:
        if assign[i] != 5 and random.random() < MODEL_MUTATION_PROB:
            assign[i] = random.randint(0, 4)
    # Repair DAG after mutation (enforce connectivity and assignment rules)
    indeg = get_indegrees(k, struct_id)
    # Connect any stray sink (non-last node with no outgoing) to last node
    bit_index = 0
    outdeg = [0]*k
    for i in range(k-1):
        for j in range(i+1, k):
            if struct_id & (1 << bit_index):
                outdeg[i] += 1
            bit_index += 1
    for node in range(k-1):
        if outdeg[node] == 0:
            idx = 0
            for a in range(node):
                idx += (k-1 - a)
            idx += (k-1 - node - 1)
            struct_id |= (1 << idx)

    # Canonicalize structure
    adj = [[0]*k for _ in range(k)]
    bit_index = 0
    for i in range(k-1):
        for j in range(i+1, k):
            if struct_id & (1 << bit_index):
                adj[i][j] = 1
            bit_index += 1
    canon_mask = canonical_representation(adj, k)
    # print("After Mutation:", canon_mask, assign)
    # Recompute indegree and fix assignments
    indeg = get_indegrees(k, canon_mask)
    # print("indegrees", indeg)
    for i in range(k):
        if indeg[i] > 1:
            assign[i] = 5
        elif indeg[i] <= 1 and assign[i] == 5:
            assign[i] = random.randint(0, 4)
    # print("After Mutation1:", struct_id, assign, k)
    return Individual(canon_mask, assign)

# Non-dominated sorting and crowding distance (NSGA-II selection mechanisms)
def fast_non_dominated_sort(population):
    fronts = [[]]
    for p in population:
        p.dom_count = 0
        p.dominated_set = []
    for p in population:
        for q in population:
            # Check domination (p dominates q?)
            if (p.objectives[0] <= q.objectives[0] and
                p.objectives[1] <= q.objectives[1] and
                p.objectives[2] <= q.objectives[2] and
                p.objectives[3] <= q.objectives[3]) and \
               (p.objectives != q.objectives):
                # p is at least as good in all objectives
                if (p.objectives[0] < q.objectives[0] or
                    p.objectives[1] < q.objectives[1] or
                    p.objectives[2] < q.objectives[2] or
                    p.objectives[3] < q.objectives[3]):
                    p.dominated_set.append(q)
            if (q.objectives[0] <= p.objectives[0] and
                q.objectives[1] <= p.objectives[1] and
                q.objectives[2] <= p.objectives[2] and
                q.objectives[3] <= p.objectives[3]) and \
               (q.objectives[0] < p.objectives[0] or
                q.objectives[1] < p.objectives[1] or
                q.objectives[2] < p.objectives[2] or
                q.objectives[3] < p.objectives[3]):
                p.dom_count += 1
        if p.dom_count == 0:
            p.rank = 0
            fronts[0].append(p)
    # Subsequent fronts
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_set:
                q.dom_count -= 1
                if q.dom_count == 0:
                    q.rank = i+1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # remove last empty front
    return fronts

def assign_crowding_distance(front):
    if not front:
        return
    n_obj = 4
    # Initialize
    for ind in front:
        ind.crowding_distance = 0
    # For each objective, sort and assign crowding distances
    for m in range(n_obj):
        front.sort(key=lambda x: x.objectives[m])
        # Extreme boundary points
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        if front[0].objectives[m] == front[-1].objectives[m]:
            continue
        # Normalize and accumulate distance for intermediate points
        for j in range(1, len(front)-1):
            dist = (front[j+1].objectives[m] - front[j-1].objectives[m]) / \
                   (front[-1].objectives[m] - front[0].objectives[m] + 1e-9)
            front[j].crowding_distance += dist

def tournament_selection(population):
    # Binary tournament: prefer lower rank, then higher crowding distance
    i, j = random.sample(range(len(population)), 2)
    a, b = population[i], population[j]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    # ranks equal, use crowding distance
    return a if a.crowding_distance >= b.crowding_distance else b

def nsga2_optimize(query_tokens, blending_prompt_tokens, ctx_tokens, df_history,
                   pop_size=100, generations=75, max_nodes=5, query_type="Sports"):
    import math

    # <<< UPDATED: on error, return worst-case (cost, latency, energy, qoa)
    def safe_eval(individual, llm_assignment):
        try:
            return evaluate_individual_V2(
                individual.struct_id,
                llm_assignment,
                query_type,
                query_tokens,
                blending_prompt_tokens,
                ctx_tokens,
                df_history
            )
        except RuntimeError as e:
            # log once
            print(f"[WARN] skipping struct_id={individual.struct_id}, assignment={llm_assignment}: {e}")
            # worst-case: infinite cost/latency/energy, zero QoA
            return math.inf, math.inf, math.inf, 0.0

    # 1) initialize
    pop = initialize_population(pop_size, max_nodes)

    # 2) evaluate initial pop
    with tqdm(pop, desc="Evaluating", unit="ind") as pbar:
      for ind in pbar:
            llm_assignment = [str(x) for x in ind.assignment]
            c, t, e, q = safe_eval(ind, llm_assignment)
            ind.metrics    = (c, t, e, q)
            ind.objectives = (c, t, e, -q)

    last_best = None
    no_improve = 0

    # 3) evolutionary loop
    with tqdm(pop, desc="Evaluating", unit="ind") as pbar:
      for gen in tqdm(range(1, generations+1), desc="time/generation", unit="gen"):
          print(f'On Generation {gen}:')
          # nondominated sort & crowding
          fronts = fast_non_dominated_sort(pop)
          for f in fronts:
              assign_crowding_distance(f)

          # produce offspring
          offspring = []
          while len(offspring) < pop_size:
              p1, p2 = tournament_selection(pop), tournament_selection(pop)
              child   = mutate(crossover(p1, p2), max_nodes)
              llm_assignment = [str(x) for x in child.assignment]
              c, t, e, q    = safe_eval(child, llm_assignment)
              child.metrics    = (c, t, e, q)
              child.objectives = (c, t, e, -q)
              offspring.append(child)

          # elitist selection
          combined = pop + offspring
          fronts   = fast_non_dominated_sort(combined)
          new_pop  = []
          f = 0
          while f < len(fronts) and len(new_pop) + len(fronts[f]) <= pop_size:
              assign_crowding_distance(fronts[f])
              new_pop.extend(fronts[f])
              f += 1
          if len(new_pop) < pop_size:
              assign_crowding_distance(fronts[f])
              fronts[f].sort(key=lambda x: x.crowding_distance, reverse=True)
              new_pop.extend(fronts[f][:pop_size - len(new_pop)])
          pop = new_pop
          best_objs = sorted([ind.objectives for ind in pop if ind.rank == 0])
          if last_best is not None and last_best >= best_objs:
              no_improve += 1
          else:
              no_improve = 0
          last_best = best_objs
          if no_improve >= 5:
              print(f'Early stop at gen {gen}')
              break

          #best_objs = sorted([ind.objectives for ind in pop if ind.rank == 0])
          #if last_best is not None and last_best >= best_objs:
          #    no_improve += 1
          #else:
          #    no_improve = 0

          #  last_best = best_objs
          #  if no_improve >= 5:
          #      print(f"Early stop at gen {gen}")
          #      break

    # return Pareto front
    return [ind for ind in pop if ind.rank == 0]



# Run optimization (example with smaller population/generations for demonstration)
query_tokens = 215
blending_prompt_tokens = 26
ctx_tokens = 39
load_from_pickle = False
#FILEPATH_TO_HISTORY_FILE = "l4.csv"
#FILEPATH_TO_HISTORY_FILE = "hist_updated.csv"
FILEPATH_TO_HISTORY_FILE = "updated_final_metrics_l4_with_tok.csv"
df_history = pd.read_csv(FILEPATH_TO_HISTORY_FILE)
df_history['llm_assignments'] = (
    df_history['llm_assignments']
      .astype(str)
      .str.replace(r'[\(\)\s]', '', regex=True)  # remove ( ) and spaces
      .str.rstrip(',')                            # remove trailing comma if any
)




args = parse_args()
if args.config:
    load_config(args.config)

# 2) Prepare to collect all results
all_results = []
print('Query types being used', QUERY_TYPES)

# 3) Loop over each query type, run NSGA 5×, and record outputs
for qt in tqdm(QUERY_TYPES, desc="Query Types"):
    for run_idx in tqdm(range(1, REPETITIONS + 1),
                        desc=f"Runs for {qt}",
                        leave=False):
        print(f"\nRunning NSGA on query type='{qt}', iteration {run_idx}/{REPETITIONS}…")
        start = time.time()

        pareto_front = nsga2_optimize(
            query_tokens,
            blending_prompt_tokens,
            ctx_tokens,
            df_history,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            max_nodes=MAX_NODES,
            query_type=qt
        )

        elapsed = time.time() - start
        print(f" → done in {elapsed:.1f}s, found {len(pareto_front)} Pareto solutions")

        # 4) For each solution on the front, pull out the metrics
        for ind in pareto_front:
            c, t, e, q = ind.metrics
            all_results.append({
                'query_type':       qt,
                'iteration':        run_idx,
                'struct_id':        ind.struct_id,
                'assignment':       ind.assignment,
                'cost':             c,
                'latency':          t,
                'energy':           e,
                'qoa':              q,
                'run_time_seconds': elapsed
            })

# # 3) Loop over each query type, run NSGA 5×, and record outputs
# for qt in query_types:
#     for run_idx in range(1, REPETITIONS + 1):
#         print(f"\nRunning NSGA on query type='{qt}', iteration {run_idx}/{REPETITIONS}…")
#         start = time.time()

#         pareto_front = nsga2_optimize(
#             query_tokens,
#             blending_prompt_tokens,
#             ctx_tokens,
#             df_history,
#             pop_size=POP_SIZE,
#             generations=GENERATIONS,
#             max_nodes=MAX_NODES,
#             query_type=qt
#         )

#         elapsed = time.time() - start
#         print(f" → done in {elapsed:.1f}s, found {len(pareto_front)} Pareto solutions")

#         # 4) For each solution on the front, pull out the metrics
#         for ind in pareto_front:
#             c, t, e, q = ind.metrics
#             all_results.append({
#                 'query_type':       qt,
#                 'iteration':        run_idx,
#                 'struct_id':        ind.struct_id,
#                 'assignment':       ind.assignment,
#                 'cost':             c,
#                 'latency':          t,
#                 'energy':           e,
#                 'qoa':              q,
#                 'run_time_seconds': elapsed
#             })

# 5) At the end, dump everything to CSV
df_out = pd.DataFrame(all_results)
out_path = "nsga_results.csv"
df_out.to_csv(RESULTS_FILE, index=False)
print(f"\nAll {len(df_out)} results saved to {RESULTS_FILE}")
# df_out.to_csv(out_path, index=False)
# print(f"\nAll results ({len(df_out)} rows) saved to {out_path}")
# python nsga_final.py --config config.json






# ============================================================================
#                  FULLY OPTIMIZED INLINE OVERLAY (Single File)
# ============================================================================

# --- Safe tqdm stub (toggle DISABLE_TQDM) ---
DISABLE_TQDM = True
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None

class _NoTQDM:
    def __init__(self, it=None, **kwargs):
        self.it = it
    def __iter__(self):
        return iter(self.it) if self.it is not None else iter(())
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def update(self, *a, **k): pass
    def set_description(self, *a, **k): pass
    def set_postfix(self, *a, **k): pass
    def close(self): pass

# Rebind tqdm used by earlier defs at runtime
tqdm = _NoTQDM if (DISABLE_TQDM or _tqdm is None) else _tqdm

# --- Shared helpers / flags ---
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any
import math, contextlib, io

NSGA_DEBUG = True
def dprint(*a, flush=False, **k):
    if NSGA_DEBUG:
        print(*a, flush=flush, **k)

# ---------------- History normalization + indexing ----------------
_HISTORY_STATE: Dict[int, Dict[str, Any]] = {}

def _normalize_assign_str(s) -> str:
    if s is None: return ""
    if isinstance(s, float):
        try:
            if math.isnan(s): return ""
        except Exception:
            pass
    if isinstance(s, str) and s.strip().lower() in {"nan", "none"}: return ""
    s = str(s).strip().replace(" ", "")
    s = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    while ",," in s: s = s.replace(",,", ",")
    if s.startswith(","): s = s[1:]
    if s.endswith(","): s = s[:-1]
    return s

def _build_history_index(df):
    import pandas as pd
    df = df.copy(deep=False)
    if "llm_assignments" in df.columns:
        df["llm_assignments"] = df["llm_assignments"].apply(_normalize_assign_str)
    else:
        df["llm_assignments"] = ""
    df["llm_list"] = df["llm_assignments"].map(lambda s: [t for t in s.split(",") if t])
    df["llm_set"]  = df["llm_list"].map(set)
    df["last_node"] = df["llm_list"].map(lambda xs: xs[-1] if xs else None)

    by_key = {}
    if {"structure_id","query_type","last_node"}.issubset(df.columns):
        for key, grp in df.groupby(["structure_id","query_type","last_node"], dropna=False):
            by_key[key] = grp

    by_inner_multiset = {}
    for _, row in df.iterrows():
        sid = row.get("structure_id"); q = row.get("query_type")
        lst = row.get("llm_list") or []
        if not lst or sid is None or q is None: continue
        sink = lst[-1]; inner = tuple(sorted(lst[:-1]))
        key = (sid, q, sink, inner)
        by_inner_multiset.setdefault(key, []).append(row)
    return {"df": df, "by_key": by_key, "by_inner_multiset": by_inner_multiset}

def _get_hist_state(df_history):
    key = id(df_history)
    st = _HISTORY_STATE.get(key)
    if st is None:
        dprint("[opt] building history index")
        st = _build_history_index(df_history)
        _HISTORY_STATE[key] = st
    return st

def clear_history_index(df_history=None):
    if df_history is None: _HISTORY_STATE.clear()
    else: _HISTORY_STATE.pop(id(df_history), None)

# ---------------- Token-level similarity helpers ----------------
FUZZY_THRESHOLD = 0.70
def _seq_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    try:
        from rapidfuzz.distance import Levenshtein as L
        return 1.0 - L.normalized_distance(a_tokens, b_tokens)
    except Exception:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a_tokens, b_tokens).ratio()

def _jaccard_set(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# ---------------- Optimized fuzzy matching ----------------
def fuzzy_ranked_matches_from_database(
    llm_assignment: List[str],
    target_structure_id: int,
    target_last_node: Optional[str],
    query_type: str,
    df_history,
    threshold: float = FUZZY_THRESHOLD,
    w_seq: float = 0.5,
    w_jac: float = 0.5,
    bonus_for_last_node: float = 0.01
):
    st = _get_hist_state(df_history)
    df_norm = st["df"]; by_key = st["by_key"]

    tgt_list = [t for t in (llm_assignment or []) if t]
    tgt_sink = target_last_node if target_last_node else (tgt_list[-1] if tgt_list else None)

    candidates = by_key.get((target_structure_id, query_type, tgt_sink))
    if candidates is None or len(candidates) == 0:
        subset = df_norm[(df_norm["structure_id"] == target_structure_id) & (df_norm["query_type"] == query_type)]
        if subset.empty: return (None, None, 0.0)
        candidates = subset

    best_list = None; best_metrics = None; best_score = 0.0
    for _, row in candidates.iterrows():
        cand_list = row["llm_list"]
        seq_sim = _seq_similarity(tgt_list, cand_list)
        jac_sim = _jaccard_set(tgt_list, cand_list)
        score = w_seq * seq_sim + w_jac * jac_sim
        if tgt_sink and cand_list and cand_list[-1] == tgt_sink: score += bonus_for_last_node
        if score > best_score:
            best_score = float(score)
            best_list = cand_list
            best_metrics = (row["cost"], row["latency"], row["energy"], row["qoa"])

    if best_list is not None and best_score >= threshold:
        return (best_list, best_metrics, best_score)
    return (None, None, best_score)

# ---------------- Exact / one-blend sub-DAG metrics ----------------
def get_subdag_metrics_v7(subdag, sub_assignment: List[str], query_type: str, df_history):
    adj_matrix = get_adj_from_graph(subdag)  # uses original helper
    structure_id = canonical_representation(adj_matrix, len(sub_assignment))
    assignment_str = _normalize_assign_str(",".join(sub_assignment))

    st = _get_hist_state(df_history); df_norm = st["df"]
    possible = df_norm[
        (df_norm["structure_id"] == structure_id) &
        (df_norm["llm_assignments"] == assignment_str) &
        (df_norm["query_type"] == query_type)
    ]
    if len(possible) == 0: return None

    cols = df_norm.columns
    def mean_if(c): return possible[c].mean() if c in cols else None
    return LLMMetrics(
        input_cost=mean_if("input_cost"),
        input_latency=mean_if("input_latency"),
        input_energy=mean_if("input_energy"),
        output_cost=mean_if("output_cost"),
        output_latency=mean_if("output_latency"),
        output_energy=mean_if("output_energy"),
        quality_of_answer=possible["qoa"].mean() if "qoa" in cols else None,
        average_output_tokens=mean_if("average_output_tokens"),
        final_cost=possible["cost"].mean() if "cost" in cols else None,
        final_latency=possible["latency"].mean() if "latency" in cols else None,
        final_energy=possible["energy"].mean() if "energy" in cols else None,
    )

def special_get_subdag_metrics_for_one_blend_operations(subdag, sub_assignment: List[str], query_type: str, df_history):
    if not sub_assignment: return None
    adj_matrix = get_adj_from_graph(subdag)
    structure_id = canonical_representation(adj_matrix, len(sub_assignment))

    st = _get_hist_state(df_history); idx = st["by_inner_multiset"]
    sink = sub_assignment[-1]; inner = tuple(sorted(sub_assignment[:-1]))
    key = (structure_id, query_type, sink, inner)
    rows = idx.get(key, []); 
    if not rows: return None

    import pandas as pd
    dfp = pd.DataFrame(rows)
    return LLMMetrics(
        input_cost=dfp["input_cost"].mean() if "input_cost" in dfp else None,
        input_latency=dfp["input_latency"].mean() if "input_latency" in dfp else None,
        input_energy=dfp["input_energy"].mean() if "input_energy" in dfp else None,
        output_cost=dfp["output_cost"].mean() if "output_cost" in dfp else None,
        output_latency=dfp["output_latency"].mean() if "output_latency" in dfp else None,
        output_energy=dfp["output_energy"].mean() if "output_energy" in dfp else None,
        quality_of_answer=dfp["qoa"].mean() if "qoa" in dfp else None,
        average_output_tokens=dfp["average_output_tokens"].mean() if "average_output_tokens" in dfp else None,
        final_cost=dfp["cost"].mean() if "cost" in dfp else None,
        final_latency=dfp["latency"].mean() if "latency" in dfp else None,
        final_energy=dfp["energy"].mean() if "energy" in dfp else None,
    )

# ---------------- Memoized evaluation wrapper ----------------
try:
    _orig_evaluate_individual_V2 = evaluate_individual_V2
    @lru_cache(maxsize=200000)
    def _eval_cache(struct_id: int, assignment_tuple: tuple, query_type: str,
                    query_tokens_tuple: tuple, blend_tokens_tuple: tuple, ctx_tokens_tuple: tuple,
                    df_key: int, cache_bust_token: Optional[int]):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return _orig_evaluate_individual_V2(
                struct_id, list(assignment_tuple), query_type,
                list(query_tokens_tuple), list(blend_tokens_tuple), list(ctx_tokens_tuple),
                _HISTORY_STATE[df_key]["df"]
            )
    def evaluate_individual_V2(struct_id, assignment, query_type, query_tokens, blending_prompt_tokens, ctx_tokens, df_history,
                               cache_bust_token: Optional[int] = None):
        df_key = id(df_history); _get_hist_state(df_history)
        return _eval_cache(int(struct_id),
                           tuple(assignment or []), str(query_type),
                           tuple(query_tokens or []), tuple(blending_prompt_tokens or []), tuple(ctx_tokens or []),
                           df_key, cache_bust_token)
except Exception:
    pass

# ---------------- Cached canonicalization + DAG gen ----------------
try:
    _orig_canonical_representation = canonical_representation
    def _adj_to_tuple(adj):
        return tuple(tuple(int(v) for v in row) for row in adj)
    @lru_cache(maxsize=20000)
    def _canonical_cached_from_tuple(adj_tup, k: int) -> int:
        adj = [list(row) for row in adj_tup]
        return _orig_canonical_representation(adj, k)
    def canonical_representation(adj_matrix, k: int) -> int:
        return _canonical_cached_from_tuple(_adj_to_tuple(adj_matrix), int(k))
except Exception:
    pass

try:
    _orig_generate_nonisomorphic_dags = generate_nonisomorphic_dags
    @lru_cache(maxsize=16)
    def generate_nonisomorphic_dags(max_nodes: int):
        dprint(f"[opt] generate_nonisomorphic_dags(k={max_nodes}) [cached]")
        return _orig_generate_nonisomorphic_dags(int(max_nodes))
except Exception:
    pass

# ---------------- Diversity utilities ----------------
def geno_key(ind) -> Tuple[int, Tuple[str, ...]]:
    return (int(ind.struct_id), tuple(ind.assignment))

def enforce_genotype_dedup(population):
    seen = set(); new_pop = []
    for ind in population:
        k = geno_key(ind)
        if k not in seen:
            seen.add(k); new_pop.append(ind)
    return new_pop

def infer_model_pool_from_history(df_history) -> List[str]:
    st = _get_hist_state(df_history); df = st["df"]
    models = set()
    for lst in df["llm_list"]:
        models.update(lst)
    return sorted(models)

def inject_random_immigrants(population, n_new: int, structures, model_pool: List[str], IndividualClass=None):
    import random
    if n_new <= 0 or not structures or not model_pool: return population
    if IndividualClass is None: IndividualClass = globals().get("Individual", None)
    for _ in range(n_new):
        s = random.choice(structures)
        if isinstance(s, (tuple, list)) and len(s) == 2:
            num_nodes, struct_id = int(s[0]), int(s[1])
        else:
            struct_id = getattr(s, "struct_id", None) or getattr(s, "id", None)
            num_nodes = getattr(s, "num_nodes", None) or getattr(s, "k", None)
            if num_nodes is None:
                try: num_nodes = len(s[0])
                except Exception: num_nodes = 3
            if struct_id is None: continue
        assign = [random.choice(model_pool) for _ in range(int(num_nodes))]
        if IndividualClass is not None:
            ind = IndividualClass(struct_id=struct_id, assignment=assign)
        else:
            class _Tmp: __slots__=("struct_id","assignment")
            def __init__(self, sid, asg): self.struct_id, self.assignment = sid, asg
            ind = _Tmp(struct_id, assign)
        population.append(ind)
    return population

class EarlyStopper:
    def __init__(self, patience: int = 8):
        self.patience = patience; self.stale = 0; self.last_keys = set()
    def update_and_should_stop(self, pareto_front) -> bool:
        keys = {(int(ind.struct_id), tuple(ind.assignment)) for ind in pareto_front}
        if keys.issubset(self.last_keys): self.stale += 1
        else: self.stale = 0; self.last_keys = keys
        return self.stale >= self.patience

def jitter_objectives(obj_tuple: Tuple[float, ...], eps: float = 1e-6) -> Tuple[float, ...]:
    import random
    return tuple(x + eps * random.random() for x in obj_tuple)


