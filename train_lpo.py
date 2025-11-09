import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import os
import time

# Import the TRL libraries (these will now work)
from trl import PPOConfig, PPOTrainer

# Import your project's custom files
from src.dataset import ProsQADataset

# --- 1. CONFIGURATION ---
# Centralized configuration for the entire script
class Config:
    # --- Project ---
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/lpo_model'
    
    # --- Model ---
    N_THOUGHTS = 6 # The number of "thought tokens" to generate
    MAX_QUESTION_LEN = 512 # Max tokens for the question
    MAX_ANSWER_LEN = 50  # Max tokens for the answer
    
    # --- PPO Training ---
    N_EPOCHS = 6
    LEARNING_RATE = 1e-5 # PPO often requires a smaller, more stable learning rate
    
    # PPO_BATCH_SIZE is for the *update* step (from the paper's 128)
    PPO_BATCH_SIZE = 128
    PPO_MINI_BATCH_SIZE = 32
    
    # ROLLOUT_BATCH_SIZE is for *generating* data
    # This must fit on your GPU.
    ROLLOUT_BATCH_SIZE = 32

# --- 2. EVALUATION FUNCTION ---
# This will evaluate our model's performance on the validation set
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating LPO Model ---")
    model.eval() # Set the model to evaluation mode
    total_correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # --- Generation ---
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=Config.N_THOUGHTS + Config.MAX_ANSWER_LEN,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract *only* the newly generated tokens
            predicted_token_ids_full = generated_ids[:, input_ids.shape[1]:]
            
            # The model's *answer* starts after its thoughts
            predicted_answer_tokens = predicted_token_ids_full[:, Config.N_THOUGHTS:]

            # --- Token-Level Accuracy Calculation ---
            num_tokens = predicted_answer_tokens.shape[1]
            label_tokens = labels[:, :num_tokens]

            valid_labels_mask = label_tokens != -100
            correct_tokens = (predicted_answer_tokens == label_tokens) & valid_labels_mask
            
            total_correct_tokens += correct_tokens.sum().item()
            total_tokens += valid_labels_mask.sum().item()

    accuracy = (total_correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"Validation Token Accuracy: {accuracy:.2f}%")
    return accuracy

# --- 3. REWARD FUNCTION (DENSE REWARD) ---
def compute_dense_reward(generated_tokens_list, label_tokens_list, tokenizer, device):
    """
    Compares generated tokens to label tokens and returns a "dense" reward
    based on token-level accuracy.
    """
    rewards = []
    
    for gen_toks, label_toks in zip(generated_tokens_list, label_tokens_list):
        # Get the same number of tokens to compare
        num_tokens = min(len(gen_toks), len(label_toks))
        
        if num_tokens == 0:
            rewards.append(torch.tensor(0.0, device=device))
            continue

        gen_toks = gen_toks[:num_tokens]
        label_toks = label_toks[:num_tokens]

        # Create a mask for non-padding tokens
        valid_labels_mask = label_toks != -100
        
        # Count correct tokens
        correct_tokens = (gen_toks == label_toks) & valid_labels_mask
        
        num_correct = correct_tokens.sum().item()
        num_total_valid = valid_labels_mask.sum().item()

        if num_total_valid > 0:
            # The reward is the percentage of correct tokens
            reward = num_correct / num_total_valid
            rewards.append(torch.tensor(reward, device=device))
        else:
            rewards.append(torch.tensor(0.0, device=device))
            
    return rewards

# --- 4. MAIN LPO TRAINING SCRIPT ---
def main():
    
    # --- Setup ---
    cfg = Config()
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- PPOConfig ---
    # This is the correct way: parameters go into the config object
    ppo_config = PPOConfig(
        learning_rate=cfg.LEARNING_RATE,
        batch_size=cfg.PPO_BATCH_SIZE,
        mini_batch_size=cfg.PPO_MINI_BATCH_SIZE,
        log_with=None # Set to None to disable logging
    )

    # --- Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: Set padding side to left for batched generation
    tokenizer.padding_side = "left"

    # --- Datasets and DataLoaders ---
    train_dataset = ProsQADataset(cfg.TRAIN_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    val_dataset = ProsQADataset(cfg.VAL_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    
    def collate_fn(batch):
        # This collator is now compatible with the new PPOTrainer dataloader
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }
    
    val_loader = DataLoader(val_dataset, batch_size=cfg.ROLLOUT_BATCH_SIZE, collate_fn=collate_fn)

    # --- Load Models ---
    # The 'policy' model we are training (with a value head)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.MODEL_NAME).to(device)
    # The 'reference' model (untrained) to stabilize RL
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.MODEL_NAME).to(device)
    
    # --- Initialize PPOTrainer ---
    # This is the correct, modern signature for TRL
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collate_fn
    )
    
    # --- Generation Settings ---
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_-token_id": tokenizer.eos_token_id,
        "max_new_tokens": cfg.N_THOUGHTS + cfg.MAX_ANSWER_LEN
    }
    
    best_accuracy = -1.0
    
    # --- LPO TRAINING LOOP ---
    for epoch in range(cfg.N_EPOCHS):
        print(f"--- Epoch {epoch+1}/{cfg.N_EPOCHS} ---")
        
        # Use the dataloader created *by* the trainer
        progress_bar = tqdm(ppo_trainer.dataloader, desc="LPO Training")
        
        for batch in progress_bar:
            query_tensors = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Convert batched tensor to a list of 1D tensors for generate
            query_list = [q for q in query_tensors]

            # 1. Act (Rollout)
            # response_tensors_list is a LIST of 1D tensors
            response_tensors_list = ppo_trainer.generate(
                query_list,
                attention_mask=attention_mask,
                **generation_kwargs
            )
            
            # --- We must manually pad responses to be the same length ---
            max_len = max(len(t) for t in response_tensors_list)
            response_tensors = []
            for t in response_tensors_list:
                padding = torch.tensor([tokenizer.pad_token_id] * (max_len - len(t)), dtype=torch.long, device=device)
                response_tensors.append(torch.cat((t, padding)))
            response_tensors = torch.stack(response_tensors)
            # --- End of padding ---

            generated_part = response_tensors[:, query_tensors.shape[1]:]

            # 2. Get Reward (Dense Reward)
            answer_tokens = generated_part[:, cfg.N_THOUGHTS:]
            
            # We need to pass lists of tensors to the reward function
            answer_tokens_list = [a for a in answer_tokens]
            labels_list = [l for l in labels]
            
            rewards = compute_dense_reward(answer_tokens_list, labels_list, tokenizer, device)
            
            # 3. Update (Learn)
            # Pass the *original* list of tensors (pre-padding) to step
            stats = ppo_trainer.step(query_list, response_tensors_list, rewards)
            
            mean_reward = torch.mean(torch.stack(rewards)).item()
            progress_bar.set_postfix({"mean_reward": f"{mean_reward:.2f}"})
        
        # --- Evaluate at the end of each epoch ---
        # We pass model.model because 'model' is the ValueHead wrapper
        accuracy = evaluate(model.model, tokenizer, val_loader, device)
        
        # --- Save the best model ---
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy! Saving model to {cfg.SAVE_PATH}")
            model.save_pretrained(cfg.SAVE_PATH)
            tokenizer.save_pretrained(cfg.SAVE_PATH)
            
    print(f"LPO Training complete! Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Baseline accuracy was: 42.95%") # Your benchmark

if __name__ == "__main__":
    main()