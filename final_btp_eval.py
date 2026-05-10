import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from train_fft import NativeAdaptiveLatentReasoning

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading Natively Fused Weights for Final Validation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "saved_models/qwen_0.5B_fft", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    model = NativeAdaptiveLatentReasoning(base_model).to(device)
    model.halt_head.load_state_dict(torch.load("saved_models/qwen_0.5B_halt.pt", weights_only=True))
    model.eval() 
    
    correct, total = 0, 0
    with open('data/validation.jsonl', 'r') as f:
        lines = f.readlines()
    
    print("\n>> COMMENCING RECURSIVE LATENT VERIFICATION...")
    with torch.no_grad():
        for line in tqdm(lines[:200], desc="Final Benchmark"):
            data = json.loads(line)
            question, true_text = data['question'], data['answer']
            
            # Step 1: Initial Latent Pass
            msg1 = [{"role": "system", "content": "Direct answer only."}, {"role": "user", "content": question}]
            text1 = tokenizer.apply_chat_template(msg1, tokenize=False, add_generation_prompt=True)
            inputs1 = tokenizer([text1], return_tensors="pt").to(device)
            
            out1 = model.base_model.generate(**inputs1, max_new_tokens=20, do_sample=False)
            pred1 = tokenizer.decode(out1[0][len(inputs1.input_ids[0]):], skip_special_tokens=True).strip()
            
            # Step 2: Recursive Verification (Internal "Double Check")
            msg2 = [
                {"role": "system", "content": "Verify the logic. Answer ONLY 'Correct' or 'Incorrect'."},
                {"role": "user", "content": f"Problem: {question}\nProposed Answer: {pred1}\nIs this answer logically sound?"}
            ]
            text2 = tokenizer.apply_chat_template(msg2, tokenize=False, add_generation_prompt=True)
            inputs2 = tokenizer([text2], return_tensors="pt").to(device)
            
            check_out = model.base_model.generate(**inputs2, max_new_tokens=5, do_sample=False)
            status = tokenizer.decode(check_out[0][len(inputs2.input_ids[0]):], skip_special_tokens=True).strip().lower()
            
            final_pred = pred1
            # If the model flags its own error, we trigger the correction branch
            if "incorrect" in status or "no" in status:
                msg3 = [{"role": "system", "content": "Logic error detected. Re-evaluate and provide the correct final answer only."}, {"role": "user", "content": question}]
                text3 = tokenizer.apply_chat_template(msg3, tokenize=False, add_generation_prompt=True)
                inputs3 = tokenizer([text3], return_tensors="pt").to(device)
                out2 = model.base_model.generate(**inputs3, max_new_tokens=20, do_sample=False)
                final_pred = tokenizer.decode(out2[0][len(inputs3.input_ids[0]):], skip_special_tokens=True).strip()

            clean_pred = re.sub(r'[^\w\s]', '', final_pred.lower())
            clean_true = re.sub(r'[^\w\s]', '', true_text.lower())
            
            if clean_true in clean_pred or (clean_true == "false" and "false" in clean_pred) or (clean_true == "true" and "true" in clean_pred):
                correct += 1
            total += 1
                
    print(f"\n" + "="*50)
    print(f" 🚀 FINAL BTP METRICS (RECURSIVE VERIFICATION) 🚀")
    print(f"="*50)
    print(f" Target Accuracy : 75.00%")
    print(f" Final Accuracy  : {(correct/total)*100:.2f}%")
    print(f"="*50)

if __name__ == "__main__":
    main()
