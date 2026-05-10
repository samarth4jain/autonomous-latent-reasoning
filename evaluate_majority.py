import torch
import re
from collections import Counter
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.dataset import ProsQADataset
from src.model_adaptive import GPT2AdaptiveLatentReasoning

def extract_logic(text):
    clean_text = re.sub(r'[^\w\s]', '', text.lower()).split()
    if "false" in clean_text:
        return "false"
    elif "true" in clean_text:
        return "true"
    elif len(clean_text) > 0:
        return clean_text[-1] # Grabs the last logical noun
    return "none"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying Massive-DPO Engine with Majority Voting on {device}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained("saved_models/baseline_model")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    val_dataset = ProsQADataset('data/validation.jsonl', tokenizer, max_q_len=400, max_a_len=50)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Load the pure, fully-restored DPO Model
    model = GPT2AdaptiveLatentReasoning.from_pretrained("saved_models/gpt2_adaptive_dpo", max_thoughts=15).to(device)
    model.halt_head.bias.data -= 1.0 # Force deep pondering
    model.eval() 
    
    correct = 0
    total = 0
    
    print("\n>> Commencing Final BTP Benchmark...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            actual_len = attention_mask[0].sum().item()
            input_ids = input_ids[:, :actual_len]
            attention_mask = attention_mask[:, :actual_len]
            
            valid_tokens = labels[0][labels[0] != -100]
            true_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
            true_logic = extract_logic(true_text)
            
            # --- MAJORITY VOTING (Self-Consistency) ---
            votes = []
            thoughts_record = []
            
            # Generate 5 alternate realities sequentially to safely track thoughts
            for _ in range(5):
                generated_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=15, 
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True, # Required for generating different paths
                    temperature=0.7,
                    top_p=0.9
                )
                
                thoughts_record.append(model.current_thoughts_used)
                
                pred_tokens = generated_ids[0, actual_len:]
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                votes.append(extract_logic(pred_text))
            
            # Count the votes and pick the winner
            vote_counts = Counter(votes)
            winning_logic = vote_counts.most_common(1)[0][0]
            avg_thoughts = sum(thoughts_record) / 5.0
            
            is_match = False
            if winning_logic == true_logic and true_logic != "none":
                is_match = True
            elif true_logic in winning_logic or winning_logic in true_logic:
                if len(true_logic) > 2: # prevent single letter false-positives
                    is_match = True

            if is_match:
                correct += 1
            
            if total < 5:
                print(f"\n--- Question {total+1} ---")
                print(f"Avg Thoughts Used : {avg_thoughts:.1f}")
                print(f"Votes Cast        : {votes}")
                print(f"Consensus Logic   : '{winning_logic}' | Expected: '{true_logic}'")
                print(f"Final Grade       : {'Correct' if is_match else 'Incorrect'}")
            
            total += 1
            if total >= 200: 
                break
                
    final_accuracy = (correct/total)*100
    print("\n" + "="*50)
    print(" 🚀 FINAL BTP DEFENSE METRICS 🚀")
    print("="*50)
    print(f" Target Accuracy  : 75.00%")
    print(f" Model Accuracy   : {final_accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
