import torch
from torch.optim import AdamW # Correct import
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

from src.model import ContinuousThoughtModel
from src.dataset import ProsQADataset

# --- EVALUATION FUNCTION (Corrected Logic) ---
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating ---")
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            if loss is not None:
                total_loss += loss.item()
            
            logits = outputs['logits']
            predicted_token_ids = torch.argmax(logits, dim=-1)
            
            num_tokens = predicted_token_ids.shape[1]
            label_tokens = labels[:, :num_tokens]

            valid_labels_mask = label_tokens != -100
            correct_batch_preds = (predicted_token_ids == label_tokens) & valid_labels_mask
            
            # Strict accuracy: all valid tokens in the sequence must match
            correct_sequences = correct_batch_preds.all(dim=1)
            correct_predictions += correct_sequences.sum().item()
            total_predictions += input_ids.shape[0]

    avg_loss = total_loss / len(val_loader)
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy (Strict): {accuracy:.2f}%")
    return accuracy

def main():
    # --- Configuration for Large Server ---
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/baseline_model'
    
    # Using full batch size as your server can handle it
    BATCH_SIZE = 64 
    
    LEARNING_RATE = 1e-4 # From the paper
    N_EPOCHS = 6 # From the paper's initial stage
    N_THOUGHTS = 6 # From the paper for logical reasoning [cite: 182-183]
    
    MAX_QUESTION_LEN = 512 # Increased from 256
    MAX_ANSWER_LEN = 50

    # --- Setup ---
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = ContinuousThoughtModel.from_pretrained(MODEL_NAME, n_thoughts=N_THOUGHTS)
    model.to(device)

    train_dataset = ProsQADataset(TRAIN_FILE, tokenizer, MAX_QUESTION_LEN, MAX_ANSWER_LEN)
    val_dataset = ProsQADataset(VAL_FILE, tokenizer, MAX_QUESTION_LEN, MAX_ANSWER_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_accuracy = -1.0

    # --- Standard Training Loop (No Gradient Accumulation) ---
    for epoch in range(N_EPOCHS):
        model.train()
        print(f"--- Epoch {epoch+1}/{N_EPOCHS} ---")
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']
            if loss is not None:
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({"loss": loss.item()})
        
        # --- Evaluate at the end of each epoch ---
        accuracy = evaluate(model, tokenizer, val_loader, device)
        
        # --- Save the best model ---
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy! Saving model to {SAVE_PATH}")
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            
    print(f"Training complete! Best validation accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()