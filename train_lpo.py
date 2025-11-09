import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

# Import the new TRL libraries
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from src.model import ContinuousThoughtModel
from src.dataset import ProsQADataset

# --- 1. REWARD FUNCTION ---
# This is a simple reward function. It will be called by the PPO trainer.
# It checks if the generated answer *exactly* matches the ground truth.
def compute_reward(model, tokenizer, val_loader, device):
    """
    This function will be used as a custom reward function.
    For simplicity in this step, we'll use a simpler version in the loop.
    Let's define a simpler reward function for the loop.
    """
    pass # We will define this inline in the main loop for clarity

# --- 2. EVALUATION FUNCTION ---
# We can reuse our evaluation function, but we must adapt it to
# use the PPO model's `generate` function correctly.
def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating LPO Model ---")
    model.eval() # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # --- Generation ---
            # We generate from the question (input_ids)
            # We must provide the correct generation parameters
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50, # Max length for an answer
                pad_token_id=tokenizer.eos_token_id
            )
            
            # --- Get Predicted Answer Tokens ---
            # The generated_ids will contain the input_ids + new_tokens
            # We only want the new_tokens
            predicted_token_ids = generated_ids[:, input_ids.shape[1]:]

            # --- Token-Level Accuracy Calculation ---
            num_tokens = predicted_token_ids.shape[1]
            # Ensure label_tokens has the same length as predicted_token_ids
            label_tokens = labels[:, :num_tokens]

            valid_labels_mask = label_tokens != -100
            
            # Handle potential shape mismatch if generation is shorter than labels
            pred_len = predicted_token_ids.shape[1]
            label_len = label_tokens.shape[1]
            
            if pred_len < label_len:
                label_tokens = label_tokens[:, :pred_len]
                valid_labels_mask = valid_labels_mask[:, :pred_len]
            elif label_len < pred_len:
                predicted_token_ids = predicted_token_ids[:, :label_len]

            correct_tokens = (predicted_token_ids == label_tokens) & valid_labels_mask
            
            total_correct_tokens += correct_tokens.sum().item()
            total_tokens += valid_labels_mask.sum().item()
            # --- End of Token-Level Accuracy ---

    accuracy = (total_correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"Validation Token Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    # --- 3. CONFIGURATION ---
    # We use the same baseline config, but add PPO-specific parameters
    config = {
        "model_name": 'gpt2',
        "train_file": 'data/train.jsonl',
        "val_file": 'data/validation.jsonl',
        "save_path": 'saved_models/lpo_model',
        "lr": 1e-5, # PPO often requires a smaller learning rate
        "n_epochs": 6,
        "n_thoughts": 6,
        "max_q_len": 512,
        "max_a_len": 50,
        "ppo_batch_size": 128, # This is the batch size for the PPO update
        "mini_batch_size": 32, # This is the mini-batch size for the PPO update
    }
    
    # PPO-specific config
    ppo_config = PPOConfig(
        #model_name=config["model_name"],
        learning_rate=config["lr"],
        batch_size=config["ppo_batch_size"],
        mini_batch_size=config["mini_batch_size"],
    )

    # --- 4. SETUP ---
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # --- 5. LOAD MODELS ---
    # We need two models for PPO:
    # 1. The 'policy' model we are training.
    # 2. A 'reference' model to stabilize training.
    
    # We load our custom ContinuousThoughtModel, but wrap it
    # with AutoModelForCausalLMWithValueHead. This wrapper
    # adds an extra "value head" needed for PPO.
    
    # IMPORTANT: We can't use our custom model class directly with TRL
    # in this simple setup. We will use a standard GPT2 model
    # and *tell it to generate extra tokens* as "thoughts".
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config["model_name"]).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config["model_name"]).to(device)
    
    # Load the datasets
    train_dataset = ProsQADataset(config["train_file"], tokenizer, config["max_q_len"], config["max_a_len"])
    val_dataset = ProsQADataset(config["val_file"], tokenizer, config["max_q_len"], config["max_a_len"])
    
    # We need a data collator to handle padding for queries
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }
    
    train_loader = DataLoader(train_dataset, batch_size=config["ppo_batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["ppo_batch_size"], collate_fn=collate_fn)

    # --- 6. INITIALIZE PPO TRAINER ---
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset, # The trainer needs the dataset to create the dataloader
        data_collator=collate_fn
    )
    
    # Generation settings for the "thoughts" + "answer"
    # We will generate N_THOUGHTS + MAX_ANSWER_LEN tokens
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": config["n_thoughts"] + config["max_a_len"]
    }
    
    best_accuracy = -1.0
    
    # --- 7. LPO TRAINING LOOP ---
    for epoch in range(config["n_epochs"]):
        print(f"--- Epoch {epoch+1}/{config['n_epochs']} ---")
        progress_bar = tqdm(ppo_trainer.dataloader, desc="LPO Training")
        
        for batch in progress_bar:
            query_tensors = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 1. Act (Rollout) - Generate thoughts + answer
            response_tensors = ppo_trainer.generate(
                query_tensors,
                attention_mask=attention_mask,
                **generation_kwargs
            )
            
            # The 'response' includes the query. We need to extract just the generated part
            generated_part = response_tensors[:, query_tensors.shape[1]:]

            # 2. Get Reward
            rewards = []
            
            # Decode labels for comparison
            label_ids = labels.clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            ground_truth_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            
            # Decode generated answers
            # We only take the *answer* part of the generation,
            # which we assume comes *after* the N_THOUGHTS
            answer_tokens = generated_part[:, config["n_thoughts"]:]
            generated_texts = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
            
            for gen_text, truth_text in zip(generated_texts, ground_truth_texts):
                gen_answer = gen_text.split(tokenizer.eos_token)[0].strip()
                truth_answer = truth_text.split(tokenizer.eos_token)[0].strip()
                
                if gen_answer == truth_answer:
                    rewards.append(torch.tensor(1.0, device=device))
                else:
                    rewards.append(torch.tensor(0.0, device=device))

            # 3. Update (Learn)
            # The ppo_trainer.step() function does all the PPO magic
            stats = ppo_trainer.step([q for q in query_tensors], [r for r in response_tensors], rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            progress_bar.set_postfix({"mean_reward": torch.mean(torch.stack(rewards)).item()})
        
        # --- Evaluate at the end of each epoch ---
        # We must pass the *policy model* (model.model) to evaluate
        accuracy = evaluate(model, tokenizer, val_loader, device)
        
        # --- Save the best model ---
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy! Saving model to {config['save_path']}")
            model.save_pretrained(config['save_path'])
            tokenizer.save_pretrained(config['save_path'])
            
    print(f"Training complete! Best validation accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()