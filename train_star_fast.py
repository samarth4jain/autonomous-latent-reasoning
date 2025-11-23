import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import random

from src.dataset import ProsQADataset
from src.model import ContinuousThoughtModel

class Config:
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/star_fast'
    
    # --- OPTIMIZATIONS FOR SPEED ---
    N_THOUGHTS = 6          # Reduced from 6
    MAX_QUESTION_LEN = 400
    MAX_ANSWER_LEN = 50
    
    N_EPOCHS = 6            # Reduced from 10
    LEARNING_RATE = 5e-6
    BATCH_SIZE = 8          # Safe batch size
    
    NUM_SAMPLES = 2         # Reduced from 8 (Massive speedup)
    TEMPERATURE = 1.0
    MAX_GRAD_NORM = 1.0
    
    # Limit how many batches to explore per epoch
    # 200 batches * 8 batch_size = 1600 examples per epoch
    MAX_STEPS_PER_EPOCH = 200 

def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating ---")
    model.eval()
    correct = 0
    total = 0
    # Limit evaluation to 50 batches to save time
    eval_limit = 50
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Eval")):
            if i >= eval_limit: break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=Config.N_THOUGHTS + Config.MAX_ANSWER_LEN,
                pad_token_id=tokenizer.eos_token_id
            )
            
            full_gen = generated_ids[:, input_ids.shape[1]:]
            gen_answer_tokens = full_gen[:, Config.N_THOUGHTS:]
            
            for j in range(gen_answer_tokens.shape[0]):
                valid_label = labels[j][labels[j] != -100]
                pred = gen_answer_tokens[j][:len(valid_label)]
                if len(pred) == len(valid_label) and torch.equal(pred, valid_label):
                    correct += 1
                total += 1
                
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc

def main():
    cfg = Config()
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = ProsQADataset(cfg.TRAIN_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    val_dataset = ProsQADataset(cfg.VAL_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    
    gen_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    model = ContinuousThoughtModel.from_pretrained(cfg.MODEL_NAME, n_thoughts=cfg.N_THOUGHTS).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    best_accuracy = -1.0

    for epoch in range(cfg.N_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{cfg.N_EPOCHS} ===")
        
        # --- PHASE 1: GENERATION ---
        print(">> Generating & Filtering samples...")
        model.eval()
        successful_examples = []
        
        # Iteration limit for speed
        for i, batch in enumerate(tqdm(gen_loader, desc="Exploration")):
            if i >= cfg.MAX_STEPS_PER_EPOCH: break 
            
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
                
                for k in range(cfg.NUM_SAMPLES):
                    sample_idx = j * cfg.NUM_SAMPLES + k
                    full_seq = gen_sequences[sample_idx]
                    pred_answer = full_seq[cfg.N_THOUGHTS : cfg.N_THOUGHTS + len(valid_label)]
                    
                    if torch.equal(pred_answer, valid_label):
                        successful_examples.append({
                            "input_ids": input_ids[j],
                            "labels": labels[j] 
                        })
                        # Optimization: If we found one good path for this question, stop checking others
                        break 

        print(f">> Found {len(successful_examples)} successful reasoning paths.")
        
        if len(successful_examples) == 0:
            print("!! No successful paths found. Continuing exploration...")
            continue

        # --- PHASE 2: TRAINING ---
        print(">> Training on successful paths...")
        model.train()
        random.shuffle(successful_examples)
        
        train_loss = 0
        steps = 0
        
        # Train on the successful data
        # We handle the list manually since it's not a DataLoader
        for i in range(0, len(successful_examples), cfg.BATCH_SIZE):
            batch_data = successful_examples[i : i + cfg.BATCH_SIZE]
            
            b_input_ids = torch.stack([x["input_ids"] for x in batch_data]).to(device)
            b_labels = torch.stack([x["labels"] for x in batch_data]).to(device)
            b_attention_mask = (b_input_ids != tokenizer.pad_token_id).long()

            optimizer.zero_grad()
            
            outputs = model(
                input_ids=b_input_ids, 
                attention_mask=b_attention_mask,
                labels=b_labels
            )
            
            loss = outputs['loss']
            
            if torch.isnan(loss):
                continue
                
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