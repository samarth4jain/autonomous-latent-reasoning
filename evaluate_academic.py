import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your custom architecture
from train_fft import NativeAdaptiveLatentReasoning

def main():
    print("\n>> [1/3] Loading Natively Fused OVERDRIVE Weights...")
    # Load tokenizer from base, but weights from our newly trained local directory
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    base = AutoModelForCausalLM.from_pretrained(
        "saved_models/qwen_academic", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    print(">> [2/3] Attaching Halt-Head...")
    model = NativeAdaptiveLatentReasoning(base).to("cuda")
    model.halt_head.load_state_dict(torch.load("saved_models/qwen_academic_halt.pt", weights_only=True))
    model.eval()

    print(">> [3/3] Running Final MMLU Benchmark...")
    correct = 0
    total = 0
    
    try:
        with open("data/mmlu_eval.jsonl", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: data/mmlu_eval.jsonl not found. Make sure you are in the project root.")
        return
        
    for line in tqdm(lines, desc="Final MMLU Benchmark"):
        data = json.loads(line)
        messages = [
            {"role": "system", "content": "Reply ONLY with A, B, C, or D."},
            {"role": "user", "content": data["question"]}
        ]
        
        # Safe tokenization bypassing the dictionary shape error
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out = model.base_model.generate(**inputs, max_new_tokens=2)
        
        # Decode only the newly generated tokens
        answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Strict exact-match logic
        if answer and answer[0].upper() == data["answer"].upper():
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    print("\n==================================================")
    print(f" 🚀 FINAL OVERDRIVE OOD METRIC (MMLU): {accuracy:.2f}%")
    print("==================================================\n")

if __name__ == "__main__":
    main()
