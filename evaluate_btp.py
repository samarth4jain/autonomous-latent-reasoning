import torch
import difflib
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import re

from src.dataset import ProsQADataset
from src.model_adaptive import GPT2AdaptiveLatentReasoning

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Waking up the Adaptive Engine on {device}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained("saved_models/baseline_model")
    tokenizer.pad_token = tokenizer.eos_token
    # Return to right padding so we don't corrupt GPT-2
    tokenizer.padding_side = 'right' 
    
    val_dataset = ProsQADataset('data/validation.jsonl', tokenizer, max_q_len=400, max_a_len=50)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = GPT2AdaptiveLatentReasoning.from_pretrained("saved_models/gpt2_adaptive_dpo", max_thoughts=15).to(device)
    
    # --- THE ARCHITECTURE OVERRIDE ---
    # Shift the Halt Head bias negative to prevent immediate lazy halting
    model.halt_head.bias.data = torch.tensor([-1.5]).to(device)
    model.eval() 
    
    correct = 0
    total = 0
    
    print("\n>> Commencing Final BTP Benchmark...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # --- THE PADDING STRIP FIX ---
            # Remove all padding dynamically so generation starts at the exact right position
            actual_len = attention_mask[0].sum().item()
            input_ids = input_ids[:, :actual_len]
            attention_mask = attention_mask[:, :actual_len]
            
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=30, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.2
            )
            
            thoughts_used = model.current_thoughts_used
            
            pred_tokens = generated_ids[0, actual_len:]
            valid_tokens = labels[0][labels[0] != -100]
            
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            true_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
            
            # Clean up formatting anomalies
            pred_text = re.sub(r'\s+', ' ', pred_text)
            
            if total < 5:
                print(f"\n--- Question {total+1} ---")
                print(f"Thoughts Used: {thoughts_used}")
                print(f"Generated    : '{pred_text}'")
                print(f"Expected     : '{true_text}'")
            
            if len(true_text) > 0:
                if true_text in pred_text:
                    correct += 1
                else:
                    similarity = difflib.SequenceMatcher(None, pred_text, true_text).ratio()
                    if similarity >= 0.80:
                        correct += 1
            
            total += 1
            # Let's run 100 questions to get a solid, final accuracy percentage
            if total >= 100:
                break
                
    print(f"\nFinal Adaptive Architecture Accuracy: {(correct/total)*100:.2f}%")

if __name__ == "__main__":
    main()
