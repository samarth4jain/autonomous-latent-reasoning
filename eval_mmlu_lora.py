import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import your custom architecture
from train_fft import NativeAdaptiveLatentReasoning

def main():
    print("\n>> [1/3] Loading Base Model & LoRA Adapters...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    base = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Load the LoRA weights we just trained
    peft_model = PeftModel.from_pretrained(base, "saved_models/qwen_academic_lora")
    
    print(">> [2/3] Attaching Halt-Head...")
    model = NativeAdaptiveLatentReasoning(peft_model).to("cuda")
    model.halt_head.load_state_dict(torch.load("saved_models/qwen_academic_halt.pt", weights_only=True))
    model.eval()

    print(">> [3/3] Running Final MMLU Benchmark...")
    correct = 0
    total = 0
    
    with open("data/mmlu_eval.jsonl", "r") as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Final MMLU Benchmark"):
        data = json.loads(line)
        messages = [
            {"role": "system", "content": "Reply ONLY with A, B, C, or D."},
            {"role": "user", "content": data["question"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out = model.base_model.generate(**inputs, max_new_tokens=2)
        
        # Decode only the newly generated tokens
        answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Exact match logic
        if answer and answer[0].upper() == data["answer"].upper():
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    print("\n==================================================")
    print(f" 🚀 FINAL LORA OOD METRIC (MMLU): {accuracy:.2f}%")
    print("==================================================\n")

if __name__ == "__main__":
    main()
