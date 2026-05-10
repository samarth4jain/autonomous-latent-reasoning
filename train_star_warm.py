import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import json
import sys

from src.dataset import ProsQADataset
from src.model import ContinuousThoughtModel

class Config:
    MODEL_PATH = 'saved_models/baseline_model' 
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    
    N_THOUGHTS = 6
    MAX_QUESTION_LEN = 400
    MAX_ANSWER_LEN = 50
    
    BATCH_SIZE = 1
    NUM_SAMPLES = 50
    TEMPERATURE = 1.4
    MATCH_THRESHOLD = 0.8 

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(cfg.MODEL_PATH)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    tokenizer.pad_token = tokenizer.eos_token

    # Load Data
    train_dataset = ProsQADataset(cfg.TRAIN_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    gen_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # Load Model
    print(f"Loading Warm Start Model from: {cfg.MODEL_PATH}")
    try:
        model = ContinuousThoughtModel.from_pretrained(cfg.MODEL_PATH, n_thoughts=cfg.N_THOUGHTS).to(device)
    except:
        print("Could not load baseline, loading base gpt2")
        model = ContinuousThoughtModel.from_pretrained('gpt2', n_thoughts=cfg.N_THOUGHTS).to(device)
    
    print("\n>> Generating & Filtering samples for DPO...")
    model.eval()
    dpo_pairs = []
    
    # --- PHASE 1: EXPLORATION AND DPO PAIRING ---
    for batch in tqdm(gen_loader, desc="Exploration"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        expanded_input_ids = input_ids.repeat_interleave(cfg.NUM_SAMPLES, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(cfg.NUM_SAMPLES, dim=0)
        
        with torch.no_grad():
            generated_ids = model.generate(
                expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=cfg.N_THOUGHTS + cfg.MAX_ANSWER_LEN,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=cfg.TEMPERATURE, 
                top_k=50
            )
        
        gen_sequences = generated_ids[:, input_ids.shape[1]:]
        
        for j in range(input_ids.shape[0]):
            label = labels[j]
            valid_label = label[label != -100]
            
            winning_path = None
            losing_path = None
            
            for k in range(cfg.NUM_SAMPLES):
                sample_idx = j * cfg.NUM_SAMPLES + k
                full_seq = gen_sequences[sample_idx]
                
                pred_answer = full_seq[cfg.N_THOUGHTS : cfg.N_THOUGHTS + len(valid_label)]
                
                if pred_answer.numel() > 0 and valid_label.numel() > 0:
                    compare_len = min(len(pred_answer), len(valid_label))
                    matches = (pred_answer[:compare_len] == valid_label[:compare_len]).sum().item()
                    accuracy = matches / len(valid_label)
                    
                    if accuracy >= cfg.MATCH_THRESHOLD and winning_path is None:
                        winning_path = full_seq
                    elif accuracy < 0.3 and losing_path is None:
                        losing_path = full_seq
            
                if winning_path is not None and losing_path is not None:
                    break 
            
            if winning_path is not None and losing_path is not None:
                dpo_pairs.append({
                    "prompt": input_ids[j].cpu().tolist(),
                    "chosen": winning_path.cpu().tolist(),
                    "rejected": losing_path.cpu().tolist()
                })

    print(f"\n>> Found {len(dpo_pairs)} complete DPO Preference Pairs.")
    
    # --- SAVE AND EXIT ---
    os.makedirs("data", exist_ok=True)
    dpo_file_path = "data/dpo_train.jsonl"
    
    print(f"Saving DPO dataset to {dpo_file_path}...")
    with open(dpo_file_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")
            
    print("Finished data generation! Stopping script safely.")
    sys.exit(0)

if __name__ == "__main__":
    main()
