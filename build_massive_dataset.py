import json
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    print("Initializing 10k Ground-Truth Distillation for Qwen-0.5B...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    dpo_pairs = []
    with open("data/validation.jsonl", "r") as f:
        lines = f.readlines()
        
    # Synthesizing 10,000 pairs to brute-force the latent pathways
    for line in tqdm(lines[:10000], desc="Synthesizing DPO Pairs"):
        data = json.loads(line)
        prompt_text = data['question']
        true_answer = data['answer']
        
        if "True" in true_answer: bad_answer = true_answer.replace("True", "False")
        elif "False" in true_answer: bad_answer = true_answer.replace("False", "True")
        else: bad_answer = true_answer[:len(true_answer)//2] + "..."
            
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        chosen_ids = tokenizer.encode(true_answer, add_special_tokens=False)
        rejected_ids = tokenizer.encode(bad_answer, add_special_tokens=False)
        
        dpo_pairs.append({"prompt": prompt_ids, "chosen": chosen_ids, "rejected": rejected_ids})
        
    with open("data/dpo_train_10k.jsonl", "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")
            
    print(f">> Saved {len(dpo_pairs)} pairs to data/dpo_train_10k.jsonl")

if __name__ == "__main__":
    main()
