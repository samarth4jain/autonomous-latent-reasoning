import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer

def main():
    print("\n>> [1/3] Initializing Combinatorial Vocabulary...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Generate massive abstract vocabularies to guarantee uniqueness
    entities = [f"Entity-{chr(65 + (i % 26))}{i}" for i in range(1000)]
    classes = [f"Class-{x}" for x in ["Alpha", "Beta", "Gamma", "Delta", "Omega", "Zeta", "Sigma", "Tau", "Phi", "Psi"]]
    classes += [f"Type-{i}" for i in range(500)]
    
    dataset = []
    target_samples = 100000
    seen_combinations = set()

    print(f">> [2/3] Synthesizing {target_samples} Unique Logical Chains...")
    
    with tqdm(total=target_samples, desc="Generating Puzzles") as pbar:
        while len(dataset) < target_samples:
            # Randomly select unique elements for the logic graph: A -> B -> C
            ent = random.choice(entities)
            c1, c2, c3 = random.sample(classes, 3)
            
            # Create a unique hash to guarantee zero duplication
            combo_hash = f"{ent}_{c1}_{c2}_{c3}"
            if combo_hash in seen_combinations:
                continue
            seen_combinations.add(combo_hash)
            
            # Build the abstract logic context
            context = f"Rule 1: All {c1} are {c2}. Rule 2: All {c2} are {c3}. Fact: {ent} is a {c1}."
            question = f"Is {ent} a {c3}?"
            
            # Format ChatML exactly how the model expects it
            messages = [
                {"role": "system", "content": "You are a logical routing engine. Deduce the answer strictly from the context."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Chosen: The perfectly routed multi-hop logic
            chosen_text = f"Yes. {ent} is a {c1}. Since all {c1} are {c2}, {ent} is a {c2}. Since all {c2} are {c3}, {ent} is a {c3}."
            
            # Rejected: A logical hallucination or failure to route
            rejected_text = f"No. {ent} is a {c1}, but there is not enough information to link it to {c3}."
            
            # Tokenize and append
            dataset.append({
                "prompt": tokenizer.encode(prompt_text, add_special_tokens=False),
                "chosen": tokenizer.encode(chosen_text, add_special_tokens=False),
                "rejected": tokenizer.encode(rejected_text, add_special_tokens=False)
            })
            pbar.update(1)

    print("\n>> [3/3] Saving Phase 2 Dataset to Disk...")
    with open("data/dpo_academic_train_100k.jsonl", "w") as f:
        for item in tqdm(dataset, desc="Writing JSONL"):
            f.write(json.dumps(item) + "\n")
            
    print("\n✅ SUCCESS: 100,000 perfectly unique training samples generated.")

if __name__ == "__main__":
    main()
