import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

class NativeAdaptiveLatentReasoning(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # THE FIX: Explicitly match the BFloat16 dtype of the base model
        self.halt_head = nn.Linear(base_model.config.hidden_size, 1, dtype=torch.bfloat16) 

    def forward(self, input_ids):
        outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
        final_hidden_state = outputs.hidden_states[-1][:, -1, :] 
        halt_logits = self.halt_head(final_hidden_state)
        return outputs.logits, halt_logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying TRUE Latent Architecture (FFT) on {device}...")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True
    )

    # THE FIX: Force the entire wrapper to respect the BFloat16 constraint
    model = NativeAdaptiveLatentReasoning(base_model).to(device).to(torch.bfloat16)
    
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = AdamW(model.parameters(), lr=1e-5)

    dpo_data = []
    # Using 1,000 pairs here so it completes in a reasonable time for your defense. 
    # (Switch to dpo_train_10k.jsonl if you want the absolute max accuracy overnight)
    with open("data/dpo_train_10k.jsonl", "r") as f:
        for line in f:
            dpo_data.append(json.loads(line))

    model.train()
    epochs = 2 
    beta = 0.1
    
    print(f"\n>> Commencing True Architecture Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for pair in tqdm(dpo_data, desc=f"Epoch {epoch+1}/{epochs}"):
            prompt_ids = torch.tensor([pair["prompt"]]).to(device)
            chosen_ids = torch.tensor([pair["chosen"]]).to(device)
            rejected_ids = torch.tensor([pair["rejected"]]).to(device)

            full_chosen = torch.cat([prompt_ids, chosen_ids], dim=1)
            full_rejected = torch.cat([prompt_ids, rejected_ids], dim=1)
            
            optimizer.zero_grad()

            logits_w, halt_w = model(input_ids=full_chosen)
            logits_l, halt_l = model(input_ids=full_rejected)

            log_probs_w = F.log_softmax(logits_w, dim=-1).mean()
            log_probs_l = F.log_softmax(logits_l, dim=-1).mean()
            pi_logratios = log_probs_w - log_probs_l
            dpo_loss = -F.logsigmoid(beta * pi_logratios)
            
            halt_prob_w = torch.sigmoid(halt_w)
            halt_prob_l = torch.sigmoid(halt_l)
            halt_loss = F.mse_loss(halt_prob_w, torch.ones_like(halt_prob_w)) + \
                        F.mse_loss(halt_prob_l, torch.zeros_like(halt_prob_l))

            total_loss = dpo_loss + (0.5 * halt_loss)
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
    print("\nTraining Complete! Saving Natively Fused Architecture...")
    model.base_model.save_pretrained("saved_models/qwen_0.5B_fft")
    torch.save(model.halt_head.state_dict(), "saved_models/qwen_0.5B_halt.pt")

if __name__ == "__main__":
    main()
