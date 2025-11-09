import torch
from torch.optim import AdamW # Correct import
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

from src.model import ContinuousThoughtModel
from src.dataset import ProsQADataset

# --- NEW: EVALUATION FUNCTION (with Token-Level Accuracy) ---
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating ---")
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    total_correct_tokens = 0
    total_tokens = 0

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
            # Get the predicted token IDs (greedy decoding)
            predicted_token_ids = torch.argmax(logits, dim=-1)
            
            # --- Token-Level Accuracy Calculation ---
            num_tokens = predicted_token_ids.shape[1]
            label_tokens = labels[:, :num_tokens]

            # Create a mask for non-padding tokens
            # We only check accuracy on tokens that are *not* -100
            valid_labels_mask = label_tokens != -100
            
            # Get correct predictions where labels are valid
            correct_tokens = (predicted_token_ids == label_tokens) & valid_labels_mask
            
            # Count the number of correct tokens and total valid tokens
            total_correct_tokens += correct_tokens.sum().item()
            total_tokens += valid_labels_mask.sum().item()
            # --- End of Token-Level Accuracy ---

    avg_loss = total_loss / len(val_loader)
    
    # Avoid division by zero if total_tokens is 0
    accuracy = (total_correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    
    print(f"Validation Loss: {avg_loss:.4f}, Validation Token Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    # --- Configuration for Large Server ---
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/baseline_model'
    
    # Using full batch size as your server can handle it
    BATCH_SIZE = 32    
    LEARNING_RATE = 1e-4 # From the paper
    N_EPOCHS = 6 # From the paper's initial stage
    N_THOUGHTS = 6 # From the paper for logical reasoning [cite: 182-183]
    
    MAX_QUESTION_LEN = 256 # Increased from 256
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