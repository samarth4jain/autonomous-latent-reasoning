import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import os

# Import the TRL libraries
from trl import PPOConfig, PPOTrainer

# Import your project's custom files
from src.dataset import ProsQADataset

# --- 1. CONFIGURATION ---
class Config:
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/lpo_model'
    
    N_THOUGHTS = 6
    MAX_QUESTION_LEN = 512
    MAX_ANSWER_LEN = 50
    
    N_EPOCHS = 6
    LEARNING_RATE = 1e-5
    
    PPO_BATCH_SIZE = 128
    PPO_MINI_BATCH_SIZE = 32
    ROLLOUT_BATCH_SIZE = 32

# --- 2. EVALUATION FUNCTION ---
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating LPO Model ---")
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=Config.N_THOUGHTS + Config.MAX_ANSWER_LEN,
                pad_token_id=tokenizer.eos_token_id
            )
            
            predicted_token_ids_full = generated_ids[:, input_ids.shape[1]:]
            predicted_answer_tokens = predicted_token_ids_full[:, Config.N_THOUGHTS:]

            num_tokens = predicted_answer_tokens.shape[1]
            label_tokens = labels[:, :num_tokens]

            valid_labels_mask = label_tokens != -100
            correct_tokens = (predicted_answer_tokens == label_tokens) & valid_labels_mask
            
            total_correct_tokens += correct_tokens.sum().item()
            total_tokens += valid_labels_mask.sum().item()

    accuracy = (total_correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"Validation Token Accuracy: {accuracy:.2f}%")
    return accuracy

# --- 3. MAIN LPO TRAINING SCRIPT ---
def main():
    
    cfg = Config()
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- PPOConfig ---
    # Parameters go into the config object
    ppo_config = PPOConfig(
        learning_rate=cfg.LEARNING_RATE,
        batch_size=cfg.PPO_BATCH_SIZE,
        mini_batch_size=cfg.PPO_MINI_BATCH_SIZE,
        log_with=None
    )

        # --- Tokenizer ---
    # --- Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # <-- ADD THIS LINE

    # --- Datasets and DataLoaders ---
    train_dataset = ProsQADataset(cfg.TRAIN_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    val_dataset = ProsQADataset(cfg.VAL_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }
    
    val_loader = DataLoader(val_dataset, batch_size=cfg.ROLLOUT_BATCH_SIZE, collate_fn=collate_fn)

    # --- Load Models ---
    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.MODEL_NAME).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.MODEL_NAME).to(device)
    
    # --- Initialize PPOTrainer ---
    # This is the correct, modern signature for TRL
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,  # This is correct for the new version
        data_collator=collate_fn
    )
    
    # --- Generation Settings ---
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": cfg.N_THOUGHTS + cfg.MAX_ANSWER_LEN
    }
    
    best_accuracy = -1.0
    
    # --- LPO TRAINING LOOP ---
    for epoch in range(cfg.N_EPOCHS):
        print(f"--- Epoch {epoch+1}/{cfg.N_EPOCHS} ---")
        
        # Use the dataloader created *by* the trainer
        progress_bar = tqdm(ppo_trainer.dataloader, desc="LPO Training")
        
        for batch in progress_bar:
            # Note: TRL's dataloader gives us lists, not tensors
            query_tensors = batch['input_ids'].to(device)
            attention_mask_tensors = batch['attention_mask'].to(device) # <-- RENAMED
            labels = batch['labels'].to(device)

            # Convert the batched tensor to a list of 1D tensors
            query_list = [q for q in query_tensors]
            mask_list = [m for m in attention_mask_tensors] # <-- ADD THIS LINE

            # 1. Act (Rollout)
            response_tensors = ppo_trainer.generate(
                query_list,
                **generation_kwargs
            )
            # 2. Get Reward (Iterate over the batch)
            rewards = []
            
            # We iterate over each item in the batch one by one
            for i in range(len(query_tensors)):
                query = query_tensors[i]
                label = labels[i]
                response = response_tensors[i] # Get the i-th tensor from the list

                # Get only the generated part of the response
                generated_part = response[len(query):]
                
                # Get the answer part (after the thoughts)
                answer_tokens = generated_part[cfg.N_THOUGHTS:]
                
                # Decode the generated answer
                gen_answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                gen_answer = gen_answer_text.split(tokenizer.eos_token)[0].strip()
                
                # Decode the ground truth answer
                label_ids = label.clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                truth_text = tokenizer.decode(label_ids, skip_special_tokens=True)
                truth_answer = truth_text.split(tokenizer.eos_token)[0].strip()

                # Compare and append reward
                if gen_answer == truth_answer:
                    rewards.append(torch.tensor(1.0, device=device))
                else:
                    rewards.append(torch.tensor(0.0, device=device))

            # 3. Update (Learn)
            # We pass the original lists to the trainer
            # Pass the attention masks for the queries to the step function
            stats = ppo_trainer.step(query_list, response_tensors, rewards, masks=mask_list)
            mean_reward = torch.mean(torch.stack(rewards)).item()
            progress_bar.set_postfix({"mean_reward": f"{mean_reward:.2f}"})
        
        # --- Evaluate at the end of each epoch ---
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