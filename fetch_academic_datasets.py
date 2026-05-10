import json
from tqdm import tqdm
from datasets import load_dataset

def fetch_mmlu_eval(output_path):
    print("\n>> Fetching MMLU (Formal Logic) Benchmark from HuggingFace...")
    
    # MMLU is fully Parquet-based and the gold standard for LLM evaluation
    ds = load_dataset("cais/mmlu", "formal_logic")
    val_data = ds['test'] 
    
    eval_pairs = []
    for item in tqdm(val_data, desc="Formatting MMLU"):
        question = item['question']
        options = item['choices']
        correct_idx = item['answer']
        
        prompt = f"Question: {question}\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}"
        answer = ["A", "B", "C", "D"][correct_idx]
        
        eval_pairs.append({"question": prompt, "answer": answer})
        
    with open(output_path, "w") as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"✅ Saved MMLU Formal Logic data to {output_path}\n")

if __name__ == "__main__":
    # We already have the 25k training data, so we only run the MMLU fetch!
    fetch_mmlu_eval("data/mmlu_eval.jsonl")
