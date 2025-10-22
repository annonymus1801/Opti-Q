import random
import itertools
import json
import os

def llm_filters():
    llms = ['0','1', '2', '3', '4']
    llm_filters = []
    for k in range(0, 5):
        for li in list(itertools.combinations(llms, k)):
            llm_filters.append(list(li))
    return llm_filters


def float_to_str(f):
    """Convert float to a safe string for filenames (e.g., 0.1 -> '0p1')."""
    return str(f).replace('.', 'p')

def slugify(text: str) -> str:
    """Convert a query type to a filesystem-safe slug."""
    return ''.join(
        c.lower() if c.isalnum() else '_'
        for c in text
    ).strip('_')

def main():
    # Parameter options
    """
    repetitions = 2
    pop_sizes = [75, 100]
    generations_list = [75, 100]
    max_nodes = 5
    prob_add_node_opts = [0.1, 0.3]
    prob_flip_edge_opts = [0.1, 0.3]
    prob_model_mutation_opts = [0.1, 0.3]
    """

    # final par1ams, subject to change
    
    """
    repetitions = 1
    pop_size = 5
    generations = 5
    max_nodes = 5
    add_p = 0.1
    flip_p = 0.1
    mut_p = 0.1
    
    """
    repetitions = 1
    pop_size = 200
    generations = 200
    max_nodes=[1,2,3,4,5]
    add_p = 0.3
    flip_p = 0.1
    mut_p = 0.3

    random.seed(52)
    # Query types to instantiate singularly
    """
    query_types = [
        'Art', 'Geography', 'History', 'Music', 'Other', 'Politics',
        'Science and technology', 'Sports', 'TV shows', 'Video games',
        'biology_mmlu', 'business_mmlu', 'chemistry_mmlu', 'computer science_mmlu',
        'economics_mmlu', 'engineering_mmlu', 'health_mmlu', 'history_mmlu',
        'law_mmlu', 'math_mmlu', 'other_mmlu', 'philosophy_mmlu',
        'physics_mmlu', 'psychology_mmlu'
    ]

    query_types = ['Art', 'Geography', 'History', 'Music', 'biology_mmlu',
       'business_mmlu', 'chemistry_mmlu', 'computer science_mmlu',
       'economics_mmlu', 'engineering_mmlu', 'health_mmlu',
       'history_mmlu', 'law_mmlu', 'math_mmlu']
"""
    query_types = ['Art', 'Geography', 'History', 'Music', 'Other', 'Politics',
       'Science and technology', 'Sports', 'TV shows', 'Video games',
       'biology_mmlu', 'business_mmlu', 'chemistry_mmlu',
       'computer science_mmlu', 'economics_mmlu', 'engineering_mmlu',
       'health_mmlu', 'history_mmlu', 'law_mmlu', 'math_mmlu',
       'other_mmlu', 'philosophy_mmlu', 'physics_mmlu', 'psychology_mmlu',
       'ALL']

    #query_types = random.sample(query_types, 10)

    # Ensure output directory exists
    os.makedirs('configs', exist_ok=True)

    total = 0
    from pprint import pprint

    for max_node in max_nodes: 
        for qt in query_types:
            qt_slug = slugify(qt)
            # Prepare safe strings
            add_str = float_to_str(add_p)
            flip_str = float_to_str(flip_p)
            mut_str = float_to_str(mut_p)

            # Build filenames
            base_name = f"{qt_slug}_pop{pop_size}_gen{generations}_add{add_str}_flip{flip_str}_mut{mut_str}_bound{max_node}"
            config_path = os.path.join('configs', f"config_{base_name}.json")
            results_filename = f"results_{base_name}.csv"

            # Build config dict
            config = {
                "repetitions": repetitions,
                "results_file_name": results_filename,
                "pop_size": pop_size,
                "generations": generations,
                "max_nodes": max_node,
                "prob_add_node": add_p,
                "prob_flip_edge": flip_p,
                "prob_model_mutation": mut_p,
                "query_types": [qt],  # single query type,
            }

            # Serialize and write JSON explicitly
            content = json.dumps(config, indent=2)
            with open(config_path, 'w') as f:
                f.write(content)

            # Verify file content
            size = os.path.getsize(config_path)
            if size == 0:
                print(f"ERROR: {config_path} is empty!")
            else:
                print(f"Generated {config_path} ({size} bytes)")

            total += 1

    print(f"\nTotal configs generated: {total}")

if __name__ == "__main__":
    main()

