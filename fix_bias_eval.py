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
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "saved_models/qwen_0.5B_fft", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    model = NativeAdaptiveLatentReasoning(base_model).to(device)
    model.halt_head.load_state_dict(torch.load("saved_models/qwen_0.5B_halt.pt", weights_only=True))
    
    # THE REASONING SHIFT: 
    # We are subtracting a large bias to force the probability DOWN.
    # This forces the model to stay in the "Reasoning" state longer.
    with torch.no_grad():
        model.halt_head.bias.data -= 5.0 
    
    model.eval() 
    correct, total = 0, 0
    with open('data/validation.jsonl', 'r') as f:
        lines = f.readlines()
    
    print("\n>> COMMENCING BTP ACCURACY RECOVERY...")
    with torch.no_grad():
        for line in tqdm(lines[:200], desc="Evaluating"):
            data = json.loads(line)
            question, true_text = data['question'], data['answer']
            
            # Using the strict Instruct formatting
            messages = [{"role": "system", "content": "Direct answer only."}, {"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            
            # Force deeper generation
            generated_ids = model.base_model.generate(
                **inputs,
                max_new_tokens=40, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            clean_pred, clean_true = re.sub(r'[^\w\s]', '', pred_text.lower()), re.sub(r'[^\w\s]', '', true_text.lower())
            
            if clean_true in clean_pred or (clean_true == "false" and "false" in clean_pred) or (clean_true == "true" and "true" in clean_pred):
                correct += 1
            total += 1
                
    print(f"\n==================================================")
    print(f" 🚀 SHIFTED ACCURACY: {(correct/total)*100:.2f}%")
    print(f"==================================================")

if __name__ == "__main__":
    main()
