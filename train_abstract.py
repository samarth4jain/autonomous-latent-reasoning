import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from train_fft import NativeAdaptiveLatentReasoning

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    model = NativeAdaptiveLatentReasoning(base_model).to(device).to(torch.bfloat16)
    for param in model.parameters(): param.requires_grad = True
        
    optimizer = AdamW(model.parameters(), lr=1e-5)

    dpo_data = []
    with open("data/dpo_abstract_25k.jsonl", "r") as f:
        for line in f: dpo_data.append(json.loads(line))

    model.train()
    print(f"\n>> TRAINING LATENT MANIFOLD (25k Abstract Samples)...")
    
    for epoch in range(2):
        for pair in tqdm(dpo_data):
            p, c, r = torch.tensor([pair["prompt"]]).to(device), torch.tensor([pair["chosen"]]).to(device), torch.tensor([pair["rejected"]]).to(device)
            full_c, full_r = torch.cat([p, c], dim=1), torch.cat([p, r], dim=1)
            
            optimizer.zero_grad()
            l_w, h_w = model(input_ids=full_c)
            l_l, h_l = model(input_ids=full_r)

            dpo_loss = -F.logsigmoid(0.1 * (F.log_softmax(l_w, dim=-1).mean() - F.log_softmax(l_l, dim=-1).mean()))
            halt_loss = F.mse_loss(torch.sigmoid(h_w), torch.ones_like(h_w)) + F.mse_loss(torch.sigmoid(h_l), torch.zeros_like(h_l))

            (dpo_loss + (0.5 * halt_loss)).backward()
            optimizer.step()
            
    model.base_model.save_pretrained("saved_models/qwen_abstract")
    torch.save(model.halt_head.state_dict(), "saved_models/qwen_abstract_halt.pt")

if __name__ == "__main__":
    main()
