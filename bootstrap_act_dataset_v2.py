import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer

def generate_distractor(classes):
    c1, c2 = random.sample(classes, 2)
    return f"Rule: All {c1} are {c2}. "

def main():
    print(">> Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    dataset = []
    entities = [f"Subject-{i}" for i in range(2000)]
    classes = [f"Class-{i}" for i in range(2000)]
    
    print(">> Generating V2 Complex Logic Chains (Balanced Yes/No)...")
    
    # 1. 1-Hop + Distractors (20k)
    for _ in tqdm(range(20000), desc="1-Hop"):
        ent, c1, c2 = random.sample(entities, 1) + random.sample(classes, 2)
        distractor = generate_distractor(classes) if random.random() > 0.5 else ""
        
        if random.random() > 0.5: # YES
            prompt = f"Context: {distractor}Fact: {ent} is a {c1}.\nQuestion: Is {ent} a {c1}?"
            dataset.append((prompt, "Yes"))
        else: # NO
            prompt = f"Context: {distractor}Fact: {ent} is a {c1}.\nQuestion: Is {ent} a {c2}?"
            dataset.append((prompt, "No"))

    # 2. 2-Hop + Distractors (30k)
    for _ in tqdm(range(30000), desc="2-Hop"):
        ent = random.choice(entities)
        c1, c2, c_fake = random.sample(classes, 3)
        distractor = generate_distractor(classes)
        
        if random.random() > 0.5: # YES
            prompt = f"Context: Rule 1: All {c1} are {c2}. {distractor}Fact: {ent} is a {c1}.\nQuestion: Is {ent} a {c2}?"
            dataset.append((prompt, "Yes"))
        else: # NO (Broken Chain)
            prompt = f"Context: Rule 1: All {c1} are {c2}. {distractor}Fact: {ent} is a {c_fake}.\nQuestion: Is {ent} a {c2}?"
            dataset.append((prompt, "No"))

    # 3. 3-Hop + Distractors (30k)
    for _ in tqdm(range(30000), desc="3-Hop"):
        ent = random.choice(entities)
        c1, c2, c3, c_fake = random.sample(classes, 4)
        distractor = generate_distractor(classes)
        
        if random.random() > 0.5: # YES
            prompt = f"Context: {distractor}Rule 1: All {c1} are {c2}. Rule 2: All {c2} are {c3}. Fact: {ent} is a {c1}.\nQuestion: Is {ent} a {c3}?"
            dataset.append((prompt, "Yes"))
        else: # NO (Missing Link)
            prompt = f"Context: {distractor}Rule 1: All {c1} are {c2}. Rule 2: All {c_fake} are {c3}. Fact: {ent} is a {c1}.\nQuestion: Is {ent} a {c3}?"
            dataset.append((prompt, "No"))

    # 4. 4-Hop Deep Logic (20k)
    for _ in tqdm(range(20000), desc="4-Hop (Deep)"):
        ent = random.choice(entities)
        c1, c2, c3, c4, c_fake = random.sample(classes, 5)
        
        if random.random() > 0.5: # YES
            prompt = f"Context: Rule 1: All {c1} are {c2}. Rule 2: All {c2} are {c3}. Rule 3: All {c3} are {c4}. Fact: {ent} is a {c1}.\nQuestion: Is {ent} a {c4}?"
            dataset.append((prompt, "Yes"))
        else: # NO
            prompt = f"Context: Rule 1: All {c1} are {c2}. Rule 2: All {c2} are {c3}. Rule 3: All {c_fake} are {c4}. Fact: {ent} is a {c1}.\nQuestion: Is {ent} a {c4}?"
            dataset.append((prompt, "No"))

    random.shuffle(dataset)
    
    print(">> Tokenizing and Saving...")
    with open("data/act_train_100k_complex.jsonl", "w") as f:
        for prompt_text, answer in tqdm(dataset, desc="Writing JSONL"):
            messages = [
                {"role": "system", "content": "You are a logical routing engine. Start with Yes or No."},
                {"role": "user", "content": prompt_text}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            p_ids = tokenizer(full_prompt).input_ids
            c_ids = tokenizer(answer).input_ids
            
            f.write(json.dumps({"prompt": p_ids, "chosen": c_ids}) + "\n")
            
    print(">> Complex Dataset V2 Generated. Ready for brutal ACT evaluation.")

if __name__ == "__main__":
    main()
