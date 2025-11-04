import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# --- IMPROVEMENT 1: INCREASED VOCABULARY DIVERSITY ---
ENTITIES = ["Alex", "Tom", "Davis", "Stella", "Jack", "Maya", "Leo", "Zoe", "Kai", "Nia", "Finn", "Eva", "Eli", "Rose", "Sam"]
CONCEPTS = [
    "gimpus", "yimpus", "worpus", "jelpus", "zhorpus", "sterpus", "lumpus", "yumpus", 
    "rempus", "fompus", "gerpus", "brimpus", "lempus", "scrompus", "rorpus", "timpus", 
    "boompus", "lorpus", "yerpus", "dumpus", "shumpus", "wumpus", "vumpus", "impus",
    "daxus", "zexus", "fipus", "mumpus", "klorpus", "jipsum", "fleekus", "narpus",
    "plumbus", "grumbo", "shleem", "drombus", "ploobus", "glaipus", "hyopian"
]

# --- IMPROVEMENT 2: VARIED GRAPH COMPLEXITY ---
def generate_prosqa_instance(min_nodes=15, max_nodes=30):
    """
    Implements Algorithm 1 from the COCONUT paper's Appendix A.2 
    to generate a single ProsQA instance with a complex logical graph.
    """
    num_nodes = random.randint(min_nodes, max_nodes)
    
    nodes = {0, 1}
    edges = []
    labels = {0: 1, 1: 2}
    groups = {0: set(), 1: {0}, 2: {1}, 3: set()}
    
    idx = 2
    while idx < num_nodes:
        rand = random.random()
        if rand <= 0.35:
            candidates = list(groups[0].union(groups[1]))
        elif rand <= 0.7:
            candidates = list(groups[0].union(groups[2]))
        else:
            candidates = list(nodes)
        
        if not candidates:
            idx += 1
            continue

        weights = [c**1.5 + 1 for c in candidates]
        n_in_nodes = min(len(candidates), np.random.poisson(1.5) + 1)
        if n_in_nodes == 0 and len(candidates) > 0:
            n_in_nodes = 1
            
        in_nodes = random.choices(candidates, weights=weights, k=n_in_nodes)
        
        cur_label = 0
        for in_idx in in_nodes:
            edges.append((in_idx, idx))
            cur_label |= labels.get(in_idx, 0)
        
        labels[idx] = cur_label
        groups[cur_label].add(idx)
        nodes.add(idx)
        idx += 1

    node_names = {}
    if len(CONCEPTS) < len(nodes):
        raise ValueError("Not enough unique concepts for the number of nodes.")
    available_concepts = random.sample(CONCEPTS, k=len(nodes))
    for node_id in nodes:
        node_names[node_id] = available_concepts.pop()
    
    entity_name = random.choice(ENTITIES)
    node_names[0] = entity_name

    out_degrees = defaultdict(int)
    for source, _ in edges:
        out_degrees[source] += 1
    leaf_nodes = [n for n in nodes if out_degrees[n] == 0]
    
    possible_targets = [n for n in leaf_nodes if labels.get(n) == 1]
    possible_distractors = [n for n in leaf_nodes if labels.get(n) == 2]

    if not possible_targets or not possible_distractors:
        return generate_prosqa_instance(min_nodes, max_nodes)
        
    target_node_id = random.choice(possible_targets)
    distractor_node_id = random.choice(possible_distractors)
    
    concept_a_true = node_names[target_node_id]
    concept_b_false = node_names[distractor_node_id]
    
    # --- IMPROVEMENT 3: DIVERSIFIED QUESTION TEMPLATES ---
    templates = [
        (f"Is {entity_name} a {concept_a_true} or {concept_b_false}?", f"{entity_name} is a {concept_a_true}."),
        (f"Is {entity_name} a {concept_b_false} or {concept_a_true}?", f"{entity_name} is a {concept_a_true}."),
        (f"True or false: {entity_name} is a {concept_a_true}.", "True."),
        (f"True or false: {entity_name} is a {concept_b_false}.", "False."),
        (f"Which of the following is correct? A) {entity_name} is a {concept_a_true}. B) {entity_name} is a {concept_b_false}.", "A"),
        (f"Which of the following is correct? A) {entity_name} is a {concept_b_false}. B) {entity_name} is a {concept_a_true}.", "B")
    ]
    question, answer = random.choice(templates)
    
    context_rules = []
    in_degrees = defaultdict(int)
    for _, target in edges:
        in_degrees[target] += 1
    root_nodes = [n for n in nodes if in_degrees[n] == 0]
    for r_node in root_nodes:
        if r_node != 0:
            context_rules.append(f"{node_names[r_node]} is a {random.choice(CONCEPTS)}.")

    for source, target in edges:
        rule = f"Every {node_names[source]} is a {node_names[target]}."
        context_rules.append(rule)
    random.shuffle(context_rules)
    
    full_question = " ".join(context_rules) + " " + question
    
    return {"question": full_question, "answer": answer}

def main():
    print("Generating large-scale ProsQA training data (this may take a few minutes)...")
    # Increased dataset size for a full-scale run
    train_data = [generate_prosqa_instance() for _ in tqdm(range(20000), desc="Training Data")]
    with open('data/train.jsonl', 'w') as f:
        for item in train_data:
            if item: f.write(json.dumps(item) + '\n')
    
    print("Generating large-scale ProsQA validation data...")
    val_data = [generate_prosqa_instance() for _ in tqdm(range(1000), desc="Validation Data")]
    with open('data/validation.jsonl', 'w') as f:
        for item in val_data:
            if item: f.write(json.dumps(item) + '\n')
    
    print("✅ Large-scale ProsQA data generation complete!")
    print("\n--- SAMPLE ---")
    print(f"Question: {train_data[0]['question']}")
    print(f"Answer: {train_data[0]['answer']}")

if __name__ == "__main__":
    main()