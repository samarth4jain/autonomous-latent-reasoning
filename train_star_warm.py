import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import random

# Import your project's custom files
from src.dataset import ProsQADataset
from src.model import ContinuousThoughtModel

class Config:
    # --- CRITICAL CHANGE: WARM START ---
    # We start with the smart model (baseline), not the dumb one (gpt2)
    MODEL_PATH = 'saved_models/baseline_model' 
    
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/star_warm_model'
    
    N_THOUGHTS = 6
    MAX_QUESTION_LEN = 400
    MAX_ANSWER_LEN = 50
    
    N_EPOCHS = 5
    # Low learning rate to gently improve the model without breaking it
    LEARNING_RATE = 5e-6 
    BATCH_SIZE = 8
    
    # Sampling: Generate 8 attempts per question
    NUM_SAMPLES = 8
    TEMPERATURE = 1.0
    MAX_GRAD_NORM = 1.0

# --- 1. EVALUATION FUNCTION (Token Accuracy) ---
# This metric allows partial credit, so you can see if the model is improving
# even if it doesn't get the full sentence perfectly right.
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating (Token Accuracy) ---")
    model.eval()
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Use greedy decoding for evaluation
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=Config.N_THOUGHTS + Config.MAX_ANSWER_LEN,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract generated answer (remove input + thoughts)
            full_gen = generated_ids[:, input_ids.shape[1]:]
            gen_answer_tokens = full_gen[:, Config.N_THOUGHTS:]
            
            # Calculate Token Accuracy
            for i in range(gen_answer_tokens.shape[0]):
                valid_label = labels[i][labels[i] != -100]
                
                # Truncate prediction to match label length for valid comparison
                pred = gen_answer_tokens[i][:len(valid_label)]
                
                # Count matching tokens
                if len(pred) == len(valid_label):
                     matches = (pred == valid_label).sum().item()
                     total_correct += matches
                     total_tokens += len(valid_label)
                elif len(pred) < len(valid_label):
                     matches = (pred == valid_label[:len(pred)]).sum().item()
                     total_correct += matches
                     total_tokens += len(valid_label)
                
    acc = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"Validation Token Accuracy: {acc:.2f}%")
    return acc

def main():
    cfg = Config()
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(cfg.MODEL_PATH)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load Data
    train_dataset = ProsQADataset(cfg.TRAIN_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    val_dataset = ProsQADataset(cfg.VAL_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    
    gen_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    # --- LOAD WARM START MODEL ---
    print(f"Loading Warm Start Model from: {cfg.MODEL_PATH}")
    # We use the custom model class to support thoughts
    model = ContinuousThoughtModel.from_pretrained(
        cfg.MODEL_PATH, 
        n_thoughts=cfg.N_THOUGHTS
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Check Baseline Accuracy
    print("Checking baseline accuracy...")
    best_accuracy = evaluate(model, tokenizer, val_loader, device)
    print(f"Starting Baseline Accuracy: {best_accuracy:.2f}%")

    for epoch in range(cfg.N_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{cfg.N_EPOCHS} ===")
        
        # --- PHASE 1: GENERATION (Self-Correction) ---
        print(">> Generating & Filtering samples...")
        model.eval()
        successful_examples = []
        
        # Check 500 batches per epoch to save time
        batches_to_check = 500
        
        for i, batch in enumerate(tqdm(gen_loader, desc="Exploration")):
            if i >= batches_to_check: break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate multiple samples per question
            expanded_input_ids = input_ids.repeat_interleave(cfg.NUM_SAMPLES, dim=0)
            expanded_attention_mask = attention_mask.repeat_interleave(cfg.NUM_SAMPLES, dim=0)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    expanded_input_ids,
                    attention_mask=expanded_attention_mask,
                    max_new_tokens=cfg.N_THOUGHTS + cfg.MAX_ANSWER_LEN,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True, # Critical for finding new paths
                    temperature=cfg.TEMPERATURE, 
                    top_k=50
                )
            
            gen_sequences = generated_ids[:, input_ids.shape[1]:]
            
            # Filter: Keep only the samples that got the answer right
            for j in range(input_ids.shape[0]):
                label = labels[j]
                valid_label = label[label != -100]
                
                for k in range(cfg.NUM_SAMPLES):
                    sample_idx = j * cfg.NUM_SAMPLES + k
                    full_seq = gen_sequences[sample_idx]
                    pred_answer = full_seq[cfg.N_THOUGHTS : cfg.N_THOUGHTS + len(valid_label)]
                    
                    if torch.equal(pred_answer, valid_label):
                        # Found a winner! Save it.
                        successful_examples.append({
                            "input_ids": input_ids[j],
                            "labels": labels[j] 
                        })
                        # Optimization: One success per question is enough
                        break 

        print(f">> Found {len(successful_examples)} successful reasoning paths.")
        
        if len(successful_examples) == 0:
            print("!! No successful paths found. This is unexpected with a warm start.")
            continue

        # --- PHASE 2: TRAINING (Fine-Tuning) ---
        print(">> Training on successful paths...")
        model.train()
        random.shuffle(successful_examples)
        
        train_loss = 0
        steps = 0
        
        for i in range(0, len(successful_examples), cfg.BATCH_SIZE):
            batch_data = successful_examples[i : i + cfg.BATCH_SIZE]
            
            if not batch_data: continue

            b_input_ids = torch.stack([x["input_ids"] for x in batch_data]).to(device)
            b_labels = torch.stack([x["labels"] for x in batch_data]).to(device)
            
            b_attention_mask = (b_input_ids != tokenizer.pad_token_id).long()

            optimizer.zero_grad()
            
            # Forward pass (calculates loss internally)
            outputs = model(
                input_ids=b_input_ids, 
                attention_mask=b_attention_mask,
                labels=b_labels
            )
            
            loss = outputs['loss']
            
            if torch.isnan(loss): continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()
            
            train_loss += loss.item()
            steps += 1
        
        avg_loss = train_loss / steps if steps > 0 else 0.0
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

        # --- PHASE 3: EVALUATION ---
        acc = evaluate(model, tokenizer, val_loader, device)
        
        if acc > best_accuracy:
            best_accuracy = acc
            print(f"New best accuracy! Saving model to {cfg.SAVE_PATH}")
            model.save_pretrained(cfg.SAVE_PATH)
            tokenizer.save_pretrained(cfg.SAVE_PATH)

    print(f"STaR Training complete! Best Acc: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()