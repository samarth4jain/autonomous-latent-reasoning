import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

# Import all your model classes
from src.model import ContinuousThoughtModel
from src.rl_model import ReinforceModel
from src.a2c_model import A2CModel
from src.dataset import ProsQADataset

class Config:
    VAL_FILE = 'data/validation.jsonl'
    N_THOUGHTS = 6
    MAX_QUESTION_LEN = 512
    MAX_ANSWER_LEN = 50
    BATCH_SIZE = 32

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_baseline(model_path, tokenizer, val_loader, device):
    print(f"--- Loading Baseline: {model_path} ---")
    try:
        model = ContinuousThoughtModel.from_pretrained(model_path, n_thoughts=Config.N_THOUGHTS).to(device)
    except:
        print(f"Could not load {model_path}. Skipping.")
        return None

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval Baseline"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Standard Generate
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=Config.N_THOUGHTS + Config.MAX_ANSWER_LEN,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract tokens (Baseline generate includes input)
            predicted_full = generated_ids[:, input_ids.shape[1]:]
            predicted_answer = predicted_full[:, Config.N_THOUGHTS:]

            # Compare
            for i in range(predicted_answer.shape[0]):
                valid_label = labels[i][labels[i] != -100]
                pred = predicted_answer[i][:len(valid_label)]
                if len(pred) == len(valid_label) and torch.equal(pred, valid_label):
                    correct += 1
                total += 1
    
    return (correct / total) * 100

def evaluate_rl(model_path, model_class, tokenizer, val_loader, device):
    print(f"--- Loading RL Model: {model_path} ---")
    try:
        model = model_class.from_pretrained(
            model_path, 
            n_thoughts=Config.N_THOUGHTS, 
            max_answer_len=Config.MAX_ANSWER_LEN
        ).to(device)
    except:
        print(f"Could not load {model_path}. Skipping.")
        return None

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval RL"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Custom Forward Generation
            # RL models return tuple, first element is answer tokens
            # A2C returns 3 or 4 values, Reinforce returns 2. 
            # We just grab the first one.
            output = model(input_ids, attention_mask)
            generated_answer = output[0]

            # Compare
            for i in range(generated_answer.shape[0]):
                valid_label = labels[i][labels[i] != -100]
                pred = generated_answer[i][:len(valid_label)]
                if len(pred) == len(valid_label) and torch.equal(pred, valid_label):
                    correct += 1
                total += 1

    return (correct / total) * 100

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load Tokenizer (Assuming baseline has the base tokenizer)
    # If this fails, use 'gpt2'
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('saved_models/baseline_model')
    except:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load Data
    val_dataset = ProsQADataset(Config.VAL_FILE, tokenizer, Config.MAX_QUESTION_LEN, Config.MAX_ANSWER_LEN)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    results = {}

    # 1. Evaluate Baseline
    acc = evaluate_baseline('saved_models/baseline_model', tokenizer, val_loader, device)
    if acc is not None: results['Baseline'] = acc

    # 2. Evaluate REINFORCE
    acc = evaluate_rl('saved_models/reinforce_model', ReinforceModel, tokenizer, val_loader, device)
    if acc is not None: results['REINFORCE'] = acc

    # 3. Evaluate A2C
    acc = evaluate_rl('saved_models/a2c_model', A2CModel, tokenizer, val_loader, device)
    if acc is not None: results['A2C'] = acc

    # 4. Evaluate A2C + BLEU
    acc = evaluate_rl('saved_models/a2c_bleu_model', A2CModel, tokenizer, val_loader, device)
    if acc is not None: results['A2C + BLEU'] = acc

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"{'Model':<20} | {'Accuracy':<10}")
    print("-" * 30)
    for name, acc in results.items():
        print(f"{name:<20} | {acc:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()