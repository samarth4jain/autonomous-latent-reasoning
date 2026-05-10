import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer

def main():
    print("Generating 25,000 Abstract Logic Pairs (Structural Reasoning)...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Abstract pools to ensure zero factual overlap with validation
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
    props = ["Glip", "Glop", "Zorp", "Blorp", "Fizz", "Buzz", "Quark", "Lepton", "Photon", "Muon"]
    
    dpo_pairs = []
    for _ in tqdm(range(25000), desc="Synthesizing"):
        n1 = random.choice(names)
        p1, p2, p3 = random.sample(props, 3)
        
        # We alternate between True and False logic chains to prevent bias
        is_true = random.choice([True, False])
        
        question = f"Statement 1: {n1} is a {p1}. Statement 2: Every {p1} is a {p2}. Statement 3: Every {p2} is a {p3}. Question: Is {n1} a {p3}?"
        
        if is_true:
            chosen = f"True. {n1} is a {p3}."
            rejected = f"False. {n1} is not a {p3}."
        else:
            # Create a false chain
            question = question.replace(f"Every {p2} is a {p3}", f"No {p2} is a {p3}")
            chosen = f"False. {n1} is not a {p3}."
            rejected = f"True. {n1} is a {p3}."
            
        p_ids = tokenizer.encode(question, add_special_tokens=False)
        c_ids = tokenizer.encode(chosen, add_special_tokens=False)
        r_ids = tokenizer.encode(rejected, add_special_tokens=False)
        
        dpo_pairs.append({"prompt": p_ids, "chosen": c_ids, "rejected": r_ids})
        
    with open("data/dpo_abstract_25k.jsonl", "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")
            
    print(">> Saved 25k pairs to data/dpo_abstract_25k.jsonl")

if __name__ == "__main__":
    main()
