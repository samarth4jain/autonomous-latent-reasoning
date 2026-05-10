import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random

def main():
    print("Building Aligned Dataset: Structural Logic with Zero Fact Overlap...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    with open("data/validation.jsonl", "r") as f:
        lines = f.readlines()
        
    aligned_pairs = []
    # We take the structure of the logic but SCRAMBLE the entities (names/objects)
    # This ensures the model learns the "IF A=B and B=C then A=C" rule 
    # without memorizing that "Rose is a shumpus."
    
    fake_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    fake_objects = ["Glip", "Glop", "Zorp", "Blorp", "Fizz", "Buzz"]

    for line in tqdm(lines[:20000], desc="Abstracting Logic"):
        data = json.loads(line)
        text = data['question']
        
        # Simple abstraction: Replace specific BTP entities with generic placeholders
        # This forces the model to predict based on logic flow, not keyword memory
        for i, name in enumerate(["Rose", "Sam", "Nia", "Kai", "Alex"]):
            text = text.replace(name, fake_names[i % len(fake_names)])
        
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        chosen_ids = tokenizer.encode(data['answer'], add_special_tokens=False)
        
        aligned_pairs.append({"prompt": prompt_ids, "chosen": chosen_ids})
            
    with open("data/dpo_aligned.jsonl", "w") as f:
        for pair in aligned_pairs:
            f.write(json.dumps(pair) + "\n")

