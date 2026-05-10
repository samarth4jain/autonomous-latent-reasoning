import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

def main():
    print("\n>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    blended_data = []
    
    print("\n>> [1/3] Loading your existing 25k Abstract Logic samples...")
    try:
        with open("data/dpo_academic_train_25k.jsonl", "r") as f:
            for line in f:
                blended_data.append(json.loads(line))
        print(f"Loaded {len(blended_data)} logic samples.")
    except FileNotFoundError:
        print("Error: Could not find the logic data. Make sure you are in the project root.")
        return

    print("\n>> [2/3] Fetching 10k English Anchors (Stanford SQuAD) to prevent forgetting...")
    # SQuAD is a reading comprehension dataset. Perfect for maintaining English parsing.
    squad = load_dataset("squad", split="train[:10000]")
    
    for item in tqdm(squad, desc="Formatting English DPO"):
        context = item["context"]
        question = item["question"]
        answer = item["answers"]["text"][0] if item["answers"]["text"] else "I don't have enough context."
        
        prompt_text = f"Context: {context}\nQuestion: {question}"
        chosen_text = answer
        rejected_text = "I don't know the answer." # Simple penalty to force engagement
        
        p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        c_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
        r_ids = tokenizer.encode(rejected_text, add_special_tokens=False)
        
        blended_data.append({"prompt": p_ids, "chosen": c_ids, "rejected": r_ids})

    print("\n>> [3/3] Shuffling the Interleaved Curriculum...")
    random.shuffle(blended_data)
    
    with open("data/dpo_blended_train.jsonl", "w") as f:
        for pair in blended_data:
            f.write(json.dumps(pair) + "\n")
            
    print(f"\n✅ SUCCESS: Saved highly robust, blended dataset with {len(blended_data)} total samples to data/dpo_blended_train.jsonl")

if __name__ == "__main__":
    main()
