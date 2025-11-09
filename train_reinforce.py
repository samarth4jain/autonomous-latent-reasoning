import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

# Import our *new* RL model and our *existing* dataset
from src.rl_model import ReinforceModel
from src.dataset import ProsQADataset

# --- 1. CONFIGURATION ---
class Config:
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/reinforce_model' # New save path
    
    N_THOUGHTS = 6
    MAX_QUESTION_LEN = 256
    MAX_ANSWER_LEN = 50    
    N_EPOCHS = 6
    LEARNING_RATE = 1e-4 # REINFORCE is sensitive, start with a small LR
    BATCH_SIZE = 16      # Use a batch size your GPU can handle

# --- 2. REWARD FUNCTION ---
def compute_reward(generated_tokens, label_tokens, tokenizer):
    """
    Compares generated tokens to label tokens and returns a reward.
    Returns a tensor of rewards (1.0 or 0.0) for each item in the batch.
    """
    device = generated_tokens.device
    batch_size = generated_tokens.shape[0]
    rewards = torch.zeros(batch_size, device=device)

    # Decode all generated answers in the batch
    gen_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    # Decode all ground-truth answers in the batch
    label_ids = label_tokens.clone()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    truth_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    for i in range(batch_size):
        gen_answer = gen_texts[i].split(tokenizer.eos_token)[0].strip()
        truth_answer = truth_texts[i].split(tokenizer.eos_token)[0].strip()
        
        if gen_answer == truth_answer:
            rewards[i] = 1.0
            
    return rewards

# --- 3. EVALUATION FUNCTION ---
# We reuse the token-level accuracy evaluator from the baseline
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating REINFORCE Model ---")
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Use our new model's generate function
            generated_answer_tokens, _ = model(input_ids, attention_mask)
            predicted_answer_tokens = generated_answer_tokens

            num_tokens = predicted_answer_tokens.shape[1]
            label_tokens = labels[:, :num_tokens]

            valid_labels_mask = label_tokens != -100
            correct_tokens = (predicted_answer_tokens == label_tokens) & valid_labels_mask
            
            total_correct_tokens += correct_tokens.sum().item()
            total_tokens += valid_labels_mask.sum().item()

    accuracy = (total_correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"Validation Token Accuracy: {accuracy:.2f}%")
    return accuracy

# --- 4. MAIN REINFORCE TRAINING SCRIPT ---
def main():
    
    cfg = Config()
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: Set padding side to left for batched generation
    tokenizer.padding_side = "left"

    # --- Load Datasets and DataLoaders ---
    train_dataset = ProsQADataset(cfg.TRAIN_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    val_dataset = ProsQADataset(cfg.VAL_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    # --- Load Model ---
    # We load our new ReinforceModel
    model = ReinforceModel.from_pretrained(
        cfg.MODEL_NAME, 
        n_thoughts=cfg.N_THOUGHTS, 
        max_answer_len=cfg.MAX_ANSWER_LEN
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    
    best_accuracy = -1.0
    
    # --- REINFORCE TRAINING LOOP ---
    for epoch in range(cfg.N_EPOCHS):
        print(f"--- Epoch {epoch+1}/{cfg.N_EPOCHS} ---")
        model.train() # Set model to training mode
        progress_bar = tqdm(train_loader, desc="REINFORCE Training")
        
        total_epoch_reward = 0.0
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # 1. Act (Rollout)
            # Get the generated answer and the log-prob of that answer
            generated_answer_tokens, total_log_prob = model(input_ids, attention_mask)

            # 2. Get Reward
            # Compare the generated answer to the ground-truth labels
            rewards = compute_reward(generated_answer_tokens, labels, tokenizer)
            
            # 3. Calculate Loss (Policy Gradient)
            # Loss is the negative of the log-probability multiplied by the reward
            # We take the mean loss across the batch
            loss = -torch.mean(total_log_prob * rewards)
            
            # 4. Update (Learn)
            loss.backward()
            optimizer.step()
            
            total_epoch_reward += rewards.mean().item()
            progress_bar.set_postfix({"mean_reward": f"{rewards.mean().item():.2f}", "loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1} average reward: {total_epoch_reward / len(train_loader):.4f}")

        # --- Evaluate at the end of each epoch ---
        accuracy = evaluate(model, tokenizer, val_loader, device)
        
        # --- Save the best model ---
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy! Saving model to {cfg.SAVE_PATH}")
            model.save_pretrained(cfg.SAVE_PATH)
            tokenizer.save_pretrained(cfg.SAVE_PATH)
            
    print(f"REINFORCE Training complete! Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Baseline accuracy was: 42.95%") # Your benchmark

if __name__ == "__main__":
    main()