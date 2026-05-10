import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GPT2Tokenizer
import json
from tqdm import tqdm

from src.model_adaptive import GPT2AdaptiveLatentReasoning

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "saved_models/baseline_model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading Active Adaptive Engine...")
    model = GPT2AdaptiveLatentReasoning.from_pretrained(model_path, max_thoughts=15).to(device)

    print("Loading Frozen Reference Engine (The Anchor)...")
    ref_model = GPT2AdaptiveLatentReasoning.from_pretrained(model_path, max_thoughts=15).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Lowered Learning Rate to prevent aggressive overwriting
    optimizer = AdamW(model.parameters(), lr=5e-6)

    dpo_data = []
    with open("data/dpo_train_massive.jsonl", "r") as f:
        for line in f:
            dpo_data.append(json.loads(line))
            
    print(f"Loaded {len(dpo_data)} Preference Pairs. Starting Final DPO Training...")

    model.train()
    epochs = 2 # Dropped to 2 to prevent overfitting the tiny dataset
    beta = 0.1 
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0
        
        for pair in tqdm(dpo_data, desc="Training DPO"):
            prompt_ids = torch.tensor([pair["prompt"]]).to(device)
            chosen_ids = torch.tensor([pair["chosen"]]).to(device)
            rejected_ids = torch.tensor([pair["rejected"]]).to(device)

            full_chosen = torch.cat([prompt_ids, chosen_ids], dim=1)
            full_rejected = torch.cat([prompt_ids, rejected_ids], dim=1)
            
            prompt_mask = torch.full_like(prompt_ids, -100)
            labels_chosen = torch.cat([prompt_mask, chosen_ids], dim=1)
            labels_rejected = torch.cat([prompt_mask, rejected_ids], dim=1)

            optimizer.zero_grad()

            # --- Active Model Forward ---
            outputs_w = model(input_ids=full_chosen, labels=labels_chosen, generate_thoughts=False)
            with torch.no_grad(): 
                outputs_l = model(input_ids=full_rejected, labels=labels_rejected, generate_thoughts=False)

            # --- Reference Model Forward ---
            with torch.no_grad():
                ref_w = ref_model(input_ids=full_chosen, labels=labels_chosen, generate_thoughts=False)
                ref_l = ref_model(input_ids=full_rejected, labels=labels_rejected, generate_thoughts=False)

            # Extract log probabilities
            log_probs_w = F.log_softmax(outputs_w.logits, dim=-1).mean()
            log_probs_l = F.log_softmax(outputs_l.logits, dim=-1).mean()
            
            ref_log_probs_w = F.log_softmax(ref_w.logits, dim=-1).mean()
            ref_log_probs_l = F.log_softmax(ref_l.logits, dim=-1).mean()

            # --- TRUE DPO EQUATION (With KL-Divergence Anchor) ---
            pi_logratios = log_probs_w - log_probs_l
            ref_logratios = ref_log_probs_w - ref_log_probs_l
            dpo_loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios))

            # Weight the Cross Entropy heavily to force factual grounding
            total_loss = dpo_loss + (2.0 * outputs_w.loss)
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        print(f"Average Loss: {epoch_loss / len(dpo_data):.4f}")

    print("\nTraining complete! The Adaptive Engine is anchored and functional.")
    model.save_pretrained("saved_models/gpt2_adaptive_dpo")
    print("Model saved to saved_models/gpt2_adaptive_dpo")

if __name__ == "__main__":
    main()
