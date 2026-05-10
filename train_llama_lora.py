import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import json
from tqdm import tqdm

class ShearedAdaptiveLatentReasoning(nn.Module):
    def __init__(self, base_model, max_thoughts=15):
        super().__init__()
        self.base_model = base_model
        self.max_thoughts = max_thoughts
        # Dynamically grabs Sheared-LLaMA's hidden size (2048) instead of hardcoding
        self.halt_head = nn.Linear(base_model.config.hidden_size, 1) 

    def forward(self, input_ids, labels=None):
        outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
        return outputs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying Sheared-LLaMA-1.3B on {device}...")

    # Princeton's open-source, non-gated model
    # Swap Sheared-LLaMA for a state-of-the-art Instruct model
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # LLaMA tokenizers often don't have a pad token set by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Base Sheared-LLaMA Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True # <--- ADD THIS LINE
    )

    print("Injecting LoRA Adapters...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    base_model = get_peft_model(base_model, lora_config)

    model = ShearedAdaptiveLatentReasoning(base_model).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    print("Loading Synthesized Dataset...")
    dpo_data = []
    with open("data/dpo_train_massive.jsonl", "r") as f:
        for line in f:
            dpo_data.append(json.loads(line))

    model.train()
    epochs = 1
    beta = 0.1
    
    print(f"\n>> Commencing LoRA DPO Training on {len(dpo_data)} pairs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for pair in tqdm(dpo_data, desc="Fine-Tuning Adaptive Logic"):
            prompt_ids = torch.tensor([pair["prompt"]]).to(device)
            chosen_ids = torch.tensor([pair["chosen"]]).to(device)
            rejected_ids = torch.tensor([pair["rejected"]]).to(device)

            full_chosen = torch.cat([prompt_ids, chosen_ids], dim=1)
            full_rejected = torch.cat([prompt_ids, rejected_ids], dim=1)
            
            optimizer.zero_grad()

            outputs_w = model(input_ids=full_chosen)
            outputs_l = model(input_ids=full_rejected)

            log_probs_w = F.log_softmax(outputs_w.logits, dim=-1).mean()
            log_probs_l = F.log_softmax(outputs_l.logits, dim=-1).mean()
            
            pi_logratios = log_probs_w - log_probs_l
            dpo_loss = -F.logsigmoid(beta * pi_logratios)
            
            dpo_loss.backward()
            optimizer.step()
            epoch_loss += dpo_loss.item()
            
    print("\nTraining Complete! Saving Architecture...")
    model.base_model.save_pretrained("saved_models/sheared_adaptive_lora")
    torch.save(model.halt_head.state_dict(), "saved_models/sheared_halt_head.pt")
    print(">> Saved. Ready for the final benchmark.")

if __name__ == "__main__":
    main()
