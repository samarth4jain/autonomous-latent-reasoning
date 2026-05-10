import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

# --- 1. Dataset Definition (SFT Next-Token Mode) ---
class LogicDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    "prompt": torch.tensor(item["prompt"], dtype=torch.long),
                    # We only need the very first token of the answer ("Yes" or "No") for the router to target
                    "target_token": torch.tensor([item["chosen"][0]], dtype=torch.long)
                })
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    prompts = torch.nn.utils.rnn.pad_sequence([item["prompt"] for item in batch], batch_first=True, padding_value=0)
    targets = torch.stack([item["target_token"] for item in batch]).squeeze(1)
    return prompts, targets

# --- 2. Adaptive Latent Architecture ---
class AdaptiveHaltHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 3-Layer MLP for complex, non-linear boundary detection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, hidden_states):
        return torch.sigmoid(self.mlp(hidden_states))

class NativeAdaptiveLatentReasoning(nn.Module):
    def __init__(self, base_model, max_latent_steps=5):
        super().__init__()
        self.base_model = base_model
        self.halt_head = AdaptiveHaltHead(base_model.config.hidden_size)
        self.max_latent_steps = max_latent_steps
        self.halting_threshold = 0.99
        
    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        
        # 1. Initial Pass: Read the English context
        outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
        current_latent_state = outputs.hidden_states[-1][:, -1:, :] 
        
        accumulated_halt_prob = torch.zeros(batch_size, 1).to(input_ids.device)
        total_ponder_penalty = 0.0
        
        # 2. The Latent Reasoning Loop
        for step in range(self.max_latent_steps):
            # Check Halt-Head confidence
            p_n = self.halt_head(current_latent_state).squeeze(-1) 
            accumulated_halt_prob += p_n
            total_ponder_penalty += accumulated_halt_prob.sum()
            
            # Break if confident
            if (accumulated_halt_prob >= self.halting_threshold).all():
                break
                
            # Otherwise, route the continuous latent thought back into the transformer
            outputs = self.base_model(inputs_embeds=current_latent_state, output_hidden_states=True)
            current_latent_state = outputs.hidden_states[-1][:, -1:, :]

        # 3. Final Output Generation (Predict the "Yes/No" token)
        final_logits = self.base_model.lm_head(current_latent_state)
        
        # We also return the final accumulated probability to anchor the loss function to 1.0
        return final_logits.squeeze(1), total_ponder_penalty, accumulated_halt_prob

# --- 3. Main Training Loop ---
def main():
    print("\n>> [1/3] Loading 1.5B Base Model for ACT SFT...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map={"": 0} 
    )

    base_model.gradient_checkpointing_enable()
    print(">> Gradient Checkpointing ENABLED.")

    print(">> [2/3] Attaching Adaptive MLP Halt-Head...")
    model = NativeAdaptiveLatentReasoning(base_model).to(device="cuda:0", dtype=torch.bfloat16)
    
    print(">> [3/3] Loading Variable-Complexity Dataset...")
    dataset = LogicDataset("data/act_train_100k_complex.jsonl") 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # --- Hyperparameters ---
    epochs = 2 
    gradient_accumulation_steps = 8 
    ponder_penalty_weight = 0.001 
    total_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    print("\n>> COMMENCING PHASE 2: ADAPTIVE LATENT ROUTING...")
    model.train()
    current_step = 0
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (prompts, targets) in enumerate(loop):
            prompts, targets = prompts.to("cuda:0"), targets.to("cuda:0")
            
            # Forward pass through the latent loop
            final_logits, ponder_penalty, final_halt_prob = model(prompts)
            
            # 1. Language Loss (Did it say Yes/No correctly?)
            lm_loss = F.cross_entropy(final_logits, targets)
            
            # 2. ACT Halt Loss (Anchor the final accumulated confidence to 1.0)
            halt_loss = F.mse_loss(final_halt_prob, torch.ones_like(final_halt_prob))
            
            # 3. Scale and apply geometric penalty
            ponder_penalty = ponder_penalty * ponder_penalty_weight
            total_loss = lm_loss + halt_loss + ponder_penalty
            total_loss = total_loss / gradient_accumulation_steps
            
            total_loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_step += 1
            
            loop.set_postfix(lm=f"{lm_loss.item():.3f}", halt=f"{halt_loss.item():.3f}", pndr=f"{ponder_penalty.item():.4f}")

    print("\n✅ ACT TRAINING COMPLETE. Saving stabilized 1.5B weights...")
    model.base_model.save_pretrained("saved_models/qwen_1.5B_act")
    torch.save(model.halt_head.state_dict(), "saved_models/qwen_1.5B_act_halt.pt")

if __name__ == "__main__":
    main()
