import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

from train_fft import NativeAdaptiveLatentReasoning

# --- 1. Dataset Definition ---
class LogicDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": torch.tensor(item["prompt"], dtype=torch.long),
            "chosen": torch.tensor(item["chosen"], dtype=torch.long),
            "rejected": torch.tensor(item["rejected"], dtype=torch.long)
        }

def collate_fn(batch):
    prompts = torch.nn.utils.rnn.pad_sequence([item["prompt"] for item in batch], batch_first=True, padding_value=0)
    chosen = torch.nn.utils.rnn.pad_sequence([item["chosen"] for item in batch], batch_first=True, padding_value=0)
    rejected = torch.nn.utils.rnn.pad_sequence([item["rejected"] for item in batch], batch_first=True, padding_value=0)
    return prompts, chosen, rejected

# --- 2. Main Training Loop ---
def main():
    print("\n>> [1/3] Loading Base Model for OVERDRIVE FFT...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    print(">> [2/3] Attaching Halt-Head...")
    model = NativeAdaptiveLatentReasoning(base_model).to("cuda")
    
    print(">> [3/3] Loading Pure Abstract Dataset...")
    dataset = LogicDataset("data/dpo_academic_train_25k.jsonl")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # --- Hyperparameter Overdrive ---
    epochs = 4
    total_steps = len(dataloader) * epochs
    
    # Differential Optimizer: Give the base model a tiny LR, Halt-Head a larger LR
    base_params = [p for n, p in model.base_model.named_parameters()]
    head_params = [p for n, p in model.halt_head.named_parameters()]
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-5}, 
        {'params': head_params, 'lr': 5e-4}  
    ])
    
    # Cosine Scheduler for smooth convergence
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    print("\n>> COMMENCING FFT OVERDRIVE (Dynamic Loss Annealing)...")
    model.train()
    current_step = 0
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for prompts, chosen, rejected in loop:
            prompts, chosen, rejected = prompts.to("cuda"), chosen.to("cuda"), rejected.to("cuda")
            
            optimizer.zero_grad()
            
            # Forward Passes
            chosen_logits, chosen_halt = model(torch.cat([prompts, chosen], dim=1))
            rejected_logits, rejected_halt = model(torch.cat([prompts, rejected], dim=1))
            
            # 1. Standard DPO Loss
            log_prob_chosen = F.log_softmax(chosen_logits, dim=-1).mean()
            log_prob_rejected = F.log_softmax(rejected_logits, dim=-1).mean()
            dpo_loss = -F.logsigmoid(0.1 * (log_prob_chosen - log_prob_rejected))
            
            # 2. Halt-Head Loss (MSE)
            halt_loss_chosen = F.mse_loss(torch.sigmoid(chosen_halt), torch.ones_like(chosen_halt))
            halt_loss_rejected = F.mse_loss(torch.sigmoid(rejected_halt), torch.zeros_like(rejected_halt))
            raw_halt_loss = halt_loss_chosen + halt_loss_rejected
            
            # 3. Dynamic Loss Annealing (Ramps from 0.0 to 0.5 over training)
            anneal_factor = min(1.0, current_step / (total_steps * 0.5))
            current_halt_weight = 0.5 * anneal_factor
            
            total_loss = dpo_loss + (current_halt_weight * raw_halt_loss)
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_step += 1
            loop.set_postfix(loss=total_loss.item(), halt_w=f"{current_halt_weight:.3f}")

    print("\n✅ OVERDRIVE COMPLETE. Saving final weights...")
    model.base_model.save_pretrained("saved_models/qwen_academic")
    torch.save(model.halt_head.state_dict(), "saved_models/qwen_academic_halt.pt")
    print(">> Run evaluate_academic.py to verify.")

if __name__ == "__main__":
    main()
