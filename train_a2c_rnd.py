import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config
from tqdm import tqdm
import os
import nltk.translate.bleu_score as bleu

from src.a2c_model import A2CModel
from src.rnd_model import RNDModel
from src.dataset import ProsQADataset

class Config:
    MODEL_NAME = 'gpt2'
    TRAIN_FILE = 'data/train.jsonl'
    VAL_FILE = 'data/validation.jsonl'
    SAVE_PATH = 'saved_models/a2c_rnd_model'
    
    N_THOUGHTS = 6
    MAX_QUESTION_LEN = 512
    MAX_ANSWER_LEN = 50
    
    N_EPOCHS = 6
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 32
    GAMMA = 0.99
    
    # RND Parameters
    # We mix Extrinsic (Task) and Intrinsic (Curiosity) rewards
    INTRINSIC_COEF = 0.5  
    ENTROPY_COEF = 0.01

def compute_bleu_reward(generated_tokens, label_tokens, tokenizer, device):
    rewards = []
    smoothing = bleu.SmoothingFunction().method1
    
    gen_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    label_ids = label_tokens.clone()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    truth_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    for i in range(len(gen_texts)):
        gen = gen_texts[i].split(tokenizer.eos_token)[0].strip().split()
        ref = [truth_texts[i].split(tokenizer.eos_token)[0].strip().split()]
        
        if len(gen) > 0:
            score = bleu.sentence_bleu(ref, gen, weights=(1,0,0,0), smoothing_function=smoothing)
            rewards.append(score)
        else:
            rewards.append(0.0)
            
    return torch.tensor(rewards, device=device)

def evaluate(model, tokenizer, val_loader, device):
    print("\n--- Evaluating ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Only need answer tokens for accuracy check
            gen_tokens, _, _, _ = model(input_ids, attention_mask)
            
            for i in range(gen_tokens.shape[0]):
                valid_label = labels[i][labels[i] != -100]
                # Take same length from prediction to compare
                pred = gen_tokens[i][:len(valid_label)]
                
                if len(pred) == len(valid_label) and torch.equal(pred, valid_label):
                    correct += 1
                total += 1
                
    acc = (correct / total) * 100
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc

def main():
    cfg = Config()
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = ProsQADataset(cfg.TRAIN_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    val_dataset = ProsQADataset(cfg.VAL_FILE, tokenizer, cfg.MAX_QUESTION_LEN, cfg.MAX_ANSWER_LEN)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    # 1. Load Agent (A2C) - FROM SCRATCH
    agent = A2CModel.from_pretrained(cfg.MODEL_NAME, n_thoughts=cfg.N_THOUGHTS, max_answer_len=cfg.MAX_ANSWER_LEN).to(device)
    
    # 2. Load Curiosity Module (RND)
    gpt_config = GPT2Config.from_pretrained(cfg.MODEL_NAME)
    rnd_model = RNDModel(vocab_size=gpt_config.vocab_size).to(device)

    # Optimizers
    optimizer_agent = AdamW(agent.parameters(), lr=cfg.LEARNING_RATE)
    optimizer_rnd = AdamW(rnd_model.predictor.parameters(), lr=cfg.LEARNING_RATE)

    best_acc = -1.0

    for epoch in range(cfg.N_EPOCHS):
        agent.train()
        rnd_model.train()
        print(f"--- Epoch {epoch+1} ---")
        
        progress = tqdm(train_loader, desc="Training")
        total_ext_reward = 0
        total_int_reward = 0
        
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer_agent.zero_grad()
            optimizer_rnd.zero_grad()

            # --- 1. Rollout ---
            # Agent generates experience
            answer_tokens, log_probs, values, all_logits = agent(input_ids, mask)

            # --- 2. Calculate Rewards ---
            # Extrinsic: BLEU score against ground truth
            reward_ext = compute_bleu_reward(answer_tokens, labels, tokenizer, device)
            
            # Intrinsic: Curiosity (Prediction Error)
            rnd_loss, reward_int = rnd_model(answer_tokens)
            
            # Combine Rewards
            total_reward = (1 - cfg.INTRINSIC_COEF) * reward_ext + (cfg.INTRINSIC_COEF * reward_int)
            
            # --- 3. Train RND ---
            # Update predictor to match target better next time
            rnd_loss.backward()
            optimizer_rnd.step()

            # --- 4. Train Agent (A2C) ---
            # Calculate Returns (R) and Advantages (A)
            returns = torch.zeros_like(values)
            advantages = torch.zeros_like(values)
            R = 0.0
            
            for t in reversed(range(cfg.MAX_ANSWER_LEN)):
                # Reward given at the last step
                r_t = total_reward if t == (cfg.MAX_ANSWER_LEN - 1) else 0.0
                R = r_t + cfg.GAMMA * R
                returns[:, t] = R
                advantages[:, t] = R - values[:, t].detach()

            # Actor Loss: -log_prob * advantage
            actor_loss = -(log_probs * advantages).mean()
            
            # Critic Loss: MSE(value_pred, return)
            critic_loss = (values - returns).pow(2).mean()
            
            # Entropy Bonus (Exploration)
            probs = torch.softmax(all_logits, dim=-1)
            log_probs_all = torch.log_softmax(all_logits, dim=-1)
            entropy = -(probs * log_probs_all).sum(dim=-1).mean()
            entropy_loss = -cfg.ENTROPY_COEF * entropy

            loss_agent = actor_loss + 0.5 * critic_loss + entropy_loss
            
            loss_agent.backward()
            optimizer_agent.step()

            # Logging
            total_ext_reward += reward_ext.mean().item()
            total_int_reward += reward_int.mean().item()
            progress.set_postfix({
                "Ext_R": f"{reward_ext.mean().item():.3f}",
                "Int_R": f"{reward_int.mean().item():.3f}",
                "Loss": f"{loss_agent.item():.3f}"
            })

        # Stats
        avg_ext = total_ext_reward / len(train_loader)
        avg_int = total_int_reward / len(train_loader)
        print(f"Avg Extrinsic Reward: {avg_ext:.4f} | Avg Intrinsic Reward: {avg_int:.4f}")

        # Evaluation
        acc = evaluate(agent, tokenizer, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            agent.save_pretrained(cfg.SAVE_PATH)
            tokenizer.save_pretrained(cfg.SAVE_PATH)
            print("Saved new best model.")

if __name__ == "__main__":
    main()