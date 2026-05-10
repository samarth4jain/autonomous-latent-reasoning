import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

from train_fft import NativeAdaptiveLatentReasoning

# --- 1. Dataset Definition (SFT Pivot) ---
class LogicDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    "prompt": torch.tensor(item["prompt"], dtype=torch.long),
                    "chosen": torch.tensor(item["chosen"], dtype=torch.long)
                    # We are completely ignoring the 'rejected' DPO path
                })
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    prompts = torch.nn.utils.rnn.pad_sequence([item["prompt"] for item in batch], batch_first=True, padding_value=0)
    chosen = torch.nn.utils.rnn.pad_sequence([item["chosen"] for item in batch], batch_first=True, padding_value=0)
    return prompts, chosen

# --- 2. Main Training Loop ---
def main():
    print("\n>> [1/3] Loading 1.5B Base Model for SFT OVERDRIVE...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    base_model.gradient_checkpointing_enable()
    print(">> Gradient Checkpointing ENABLED.")

    print(">> [2/3] Attaching Halt-Head...")
    model = NativeAdaptiveLatentReasoning(base_model).to("cuda")
    
    print(">> [3/3] Loading 100k Dataset (Supervised Target Mode)...")
    dataset = LogicDataset("data/dpo_academic_train_100k.jsonl")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # --- Hyperparameters ---
    epochs = 2 
    gradient_accumulation_steps = 8 
    total_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    print("\n>> COMMENCING PHASE 2: SFT LATENT STABILIZATION...")
    model.train()
    current_step = 0
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (prompts, chosen) in enumerate(loop):
            prompts, chosen = prompts.to("cuda"), chosen.to("cuda")
            
            # Create full input sequence
            input_ids = torch.cat([prompts, chosen], dim=1)
            
            # Create labels (masking out the prompt so loss is only calculated on the answer)
            labels = input_ids.clone()
            labels[:, :prompts.shape[1]] = -100 
            
            # Forward Pass
            logits, halt_logits = model(input_ids)
            
            # 1. Standard Causal Language Modeling Loss (SFT)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 2. Halt-Head Loss (Anchored strictly to 1.0/True for chosen paths)
            halt_loss = F.mse_loss(torch.sigmoid(halt_logits), torch.ones_like(halt_logits))
            
            # Dynamic Annealing
            anneal_factor = min(1.0, current_step / (total_steps * 0.5))
            current_halt_weight = 0.5 * anneal_factor
            
            total_loss = lm_loss + (current_halt_weight * halt_loss)
            
            # Scale and Accumulate
            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                current_step += 1
            
            loop.set_postfix(lm_loss=f"{lm_loss.item():.4f}", halt_w=f"{current_halt_weight:.3f}")

    print("\n✅ SFT COMPLETE. Saving stabilized 1.5B weights...")
    model.base_model.save_pretrained("saved_models/qwen_1.5B_sft")
    torch.save(model.halt_head.state_dict(), "saved_models/qwen_1.5B_sft_halt.pt")

if __name__ == "__main__":
    main()
