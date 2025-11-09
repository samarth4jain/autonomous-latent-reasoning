import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import nltk.translate.bleu_score as bleu

# Import our A2C model and dataset
from src.a2c_model import A2CModel
from src.dataset import ProsQADataset

# --- 1. CONFIGURATION ---
class Config:
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/a2c_bleu_model' # New save path
    
    N_THOUGHTS = 6
    MAX_QUESTION_LEN = 256
    MAX_ANSWER_LEN = 50
    
    N_EPOCHS = 6
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 16
    GAMMA = 0.99         # Discount factor
    ENTROPY_COEF = 0.01  # Coefficient for the entropy bonus

# --- 2. REWARD FUNCTION (BLEU SCORE) ---
def compute_bleu_reward(generated_tokens, label_tokens, tokenizer):
    """
    Compares generated tokens to label tokens using BLEU score for a dense reward.
    """
    device = generated_tokens.device
    batch_size = generated_tokens.shape[0]
    rewards = torch.zeros(batch_size, device=device)
    
    # Use a simple 1-gram BLEU (token overlap)
    smoothing_fn = bleu.SmoothingFunction().method1

    # Decode all in batch
    gen_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    label_ids = label_tokens.clone()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    truth_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    for i in range(batch_size):
        gen_answer = gen_texts[i].split(tokenizer.eos_token)[0].strip()
        truth_answer = truth_texts[i].split(tokenizer.eos_token)[0].strip()
        
        # Tokenize for BLEU
        hypothesis = gen_answer.split()
        reference = [truth_answer.split()] # BLEU expects a list of references
        
        if len(hypothesis) > 0:
            # Calculate 1-gram BLEU score
            bleu_score = bleu.sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing_fn)
            rewards[i] = bleu_score
        
    return rewards

# --- 3. EVALUATION FUNCTION (Token-Level Accuracy) ---
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating A2C Model ---")
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Use our model's generate function
            generated_answer_tokens, _, _ = model(input_ids, attention_mask)
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

# --- 4. MAIN A2C TRAINING SCRIPT ---
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
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    model = A2CModel.from_pretrained(
        cfg.MODEL_NAME, # Train from scratch
        n_thoughts=cfg.N_THOUGHTS, 
        max_answer_len=cfg.MAX_ANSWER_LEN
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    best_accuracy = -1.0
    
    # --- A2C TRAINING LOOP ---
    for epoch in range(cfg.N_EPOCHS):
        print(f"--- Epoch {epoch+1}/{cfg.N_EPOCHS} ---")
        model.train()
        progress_bar = tqdm(train_loader, desc="A2C+BLEU Training")
        
        total_epoch_reward = 0.0
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # 1. Act & Criticize (Rollout)
            answer_tokens, log_probs, values, all_logits = model(input_ids, attention_mask)

            # 2. Get Final Reward (using BLEU)
            final_rewards = compute_bleu_reward(answer_tokens, labels, tokenizer)
            total_epoch_reward += final_rewards.mean().item()
            
            # 3. Calculate Returns and Advantages
            returns = torch.zeros_like(values)
            advantages = torch.zeros_like(values)
            R = 0.0
            for t in reversed(range(cfg.MAX_ANSWER_LEN)):
                step_reward = final_rewards if t == cfg.MAX_ANSWER_LEN - 1 else 0.0
                R = step_reward + cfg.GAMMA * R
                returns[:, t] = R
                advantages[:, t] = R - values[:, t]

            # 4. Calculate Loss
            actor_loss = -torch.mean(log_probs * advantages.detach())
            critic_loss = torch.mean(advantages.pow(2))
            
            # --- NEW: ENTROPY BONUS ---
            # This encourages exploration
            probs = torch.softmax(all_logits, dim=-1)
            log_probs_dist = torch.log_softmax(all_logits, dim=-1)
            entropy = -torch.sum(probs * log_probs_dist, dim=-1)
            entropy_loss = -cfg.ENTROPY_COEF * torch.mean(entropy)
            
            # Total Loss
            total_loss = actor_loss + 0.5 * critic_loss + entropy_loss # Add entropy
            
            # 5. Update (Learn)
            total_loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({
                "mean_reward": f"{final_rewards.mean().item():.3f}", 
                "loss": f"{total_loss.item():.4f}"
            })
        
        print(f"Epoch {epoch+1} average reward: {total_epoch_reward / len(train_loader):.4f}")

        # --- Evaluate at the end of each epoch ---
        accuracy = evaluate(model, tokenizer, val_loader, device)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy! Saving model to {cfg.SAVE_PATH}")
            model.save_pretrained(cfg.SAVE_PATH)
            tokenizer.save_pretrained(cfg.SAVE_PATH)
            
    print(f"A2C+BLEU Training complete! Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Baseline accuracy was: 42.95%")

if __name__ == "__main__":
    main()