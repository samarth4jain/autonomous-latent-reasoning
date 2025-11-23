import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import os

from src.dataset import ProsQADataset

class Config:
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/star_model'
    
    N_THOUGHTS = 6
    MAX_QUESTION_LEN = 512
    MAX_ANSWER_LEN = 50
    
    # STaR Parameters
    N_EPOCHS = 10          # More epochs because we build data iteratively
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 8         # Keep small for generation
    NUM_SAMPLES = 8        # How many attempts per question (Exploration width)
    TEMPERATURE = 1.0      # High temp to encourage diversity

# --- 1. EVALUATION FUNCTION ---
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating STaR Model ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Greedy decoding for evaluation
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=Config.N_THOUGHTS + Config.MAX_ANSWER_LEN,
                pad_token_id=tokenizer.eos_token_id
            )
            
            full_gen = generated_ids[:, input_ids.shape[1]:]
            # Answer is after thoughts
            gen_answer_tokens = full_gen[:, Config.N_THOUGHTS:]
            
            for i in range(gen_answer_tokens.shape[0]):
                valid_label = labels[i][labels[i] != -100]
                pred = gen_answer_tokens[i][:len(valid_label)]
                
                if len(pred) == len(valid_label) and torch.equal(pred, valid_label):
                    correct += 1
                total += 1
                
    acc = (correct / total) * 100
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc

# --- 2. MAIN STaR TRAINING LOOP ---
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
    
    # Loader for GENERATION (finding good paths)
    gen_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    # Load Model (From Scratch)
    # We use the custom model class to support the N_THOUGHTS logic
    # but we treat it as a standard LM for generation
    from src.model import ContinuousThoughtModel
    model = ContinuousThoughtModel.from_pretrained(cfg.MODEL_NAME, n_thoughts=cfg.N_THOUGHTS).to(device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    best_accuracy = -1.0

    for epoch in range(cfg.N_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{cfg.N_EPOCHS} ===")
        
        # --- PHASE 1: GENERATION & FILTERING (Self-Correction) ---
        print(">> Generating & Filtering samples...")
        model.eval() # Use eval mode for generation
        
        successful_examples = [] # Store (input, successful_output) pairs
        
        for batch in tqdm(gen_loader, desc="Exploration"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # We generate multiple samples per question to find a good one
            # We expand the inputs to match NUM_SAMPLES
            # (batch * num_samples, seq_len)
            expanded_input_ids = input_ids.repeat_interleave(cfg.NUM_SAMPLES, dim=0)
            expanded_attention_mask = attention_mask.repeat_interleave(cfg.NUM_SAMPLES, dim=0)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    expanded_input_ids,
                    attention_mask=expanded_attention_mask,
                    max_new_tokens=cfg.N_THOUGHTS + cfg.MAX_ANSWER_LEN,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,          # Enable sampling for diversity
                    temperature=cfg.TEMPERATURE, 
                    top_k=50
                )
            
            # Check which samples are correct
            gen_sequences = generated_ids[:, input_ids.shape[1]:] # Thoughts + Answer
            
            for i in range(input_ids.shape[0]): # Iterate over original batch
                label = labels[i]
                valid_label = label[label != -100]
                
                # Check the K samples for this specific question
                for k in range(cfg.NUM_SAMPLES):
                    sample_idx = i * cfg.NUM_SAMPLES + k
                    full_seq = gen_sequences[sample_idx]
                    
                    # Extract answer part (after thoughts)
                    pred_answer = full_seq[cfg.N_THOUGHTS : cfg.N_THOUGHTS + len(valid_label)]
                    
                    # If correct, save this "Thought -> Answer" path
                    if torch.equal(pred_answer, valid_label):
                        # We construct a new training label:
                        # -100 for question
                        # Keep thoughts + answer tokens
                        
                        # Reconstruct the full successful sequence for training
                        good_input = input_ids[i]
                        good_output = full_seq # (Thoughts + Answer)
                        
                        successful_examples.append({
                            "input_ids": good_input,
                            "output_ids": good_output
                        })
                        
                        # If we found one good path, that's enough for this question?
                        # Let's keep all valid paths to reinforce diverse reasoning.
                        pass

        print(f">> Found {len(successful_examples)} successful reasoning paths.")
        
        if len(successful_examples) == 0:
            print("!! No successful paths found this epoch. Skipping training step.")
            continue

        # --- PHASE 2: TRAINING (Supervised Fine-Tuning on Self-Generated Data) ---
        print(">> Training on successful paths...")
        model.train()
        
        # Create a temporary dataloader for the successful examples
        # We shuffle them to break correlation
        random.shuffle(successful_examples)
        
        # Custom micro-batching loop
        train_loss = 0
        steps = 0
        
        # Process in chunks of BATCH_SIZE
        for i in range(0, len(successful_examples), cfg.BATCH_SIZE):
            batch_data = successful_examples[i : i + cfg.BATCH_SIZE]
            
            # Stack tensors
            b_input_ids = torch.stack([x["input_ids"] for x in batch_data]).to(device)
            b_output_ids = torch.stack([x["output_ids"] for x in batch_data]).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # We need to pass the thoughts+answer as "labels" to the model?
            # The ContinuousThoughtModel logic in src/model.py expects 'labels' to be just the answer.
            # BUT here we want to train on the *Thoughts* too (because they led to success).
            
            # To train on thoughts + answer, we treat this as standard causal LM training.
            # We need to bypass the ContinuousThoughtModel's special logic and just use the underlying GPT2.
            # Or, we adapt.
            
            # Simplest way: Use standard GPT2 training on the concatenated sequence.
            # Question + Thoughts + Answer
            
            # Construct full sequence
            # b_input_ids: [Q]
            # b_output_ids: [Thoughts, A]
            
            full_seq = torch.cat((b_input_ids, b_output_ids), dim=1)
            
            # Create labels: mask Question
            labels = full_seq.clone()
            labels[:, :b_input_ids.shape[1]] = -100 
            
            # Standard GPT-2 forward pass (bypassing the 'n_thoughts' loop)
            # We access the internal transformer directly or use standard inputs
            outputs = model(input_ids=full_seq, labels=labels)
            
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            steps += 1
        
        print(f"Epoch {epoch+1} Training Loss: {train_loss/steps:.4f}")

        # --- PHASE 3: EVALUATION ---
        acc = evaluate(model, tokenizer, val_loader, device)
        
        if acc > best_accuracy:
            best_accuracy = acc
            print(f"New best accuracy! Saving model to {cfg.SAVE_PATH}")
            model.save_pretrained(cfg.SAVE_PATH)
            tokenizer.save_pretrained(cfg.SAVE_PATH)

    print(f"STaR Training complete! Best validation accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    import random # Ensure random is imported
    main()