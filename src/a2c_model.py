import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class A2CModel(GPT2LMHeadModel):
    def __init__(self, config, n_thoughts, max_answer_len):
        super().__init__(config)
        self.n_thoughts = n_thoughts
        self.max_answer_len = max_answer_len
        
        # Critic Head: Estimates the Value (V) of the state
        self.critic_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 1. Process Question
        inputs_embeds = self.transformer.wte(input_ids)
        current_embeds = inputs_embeds
        current_attention_mask = attention_mask
        ones_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        past_key_values = None

        # 2. Generate Latent Thoughts
        for _ in range(self.n_thoughts):
            outputs = self.transformer(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            last_hidden_state = outputs.last_hidden_state[:, -1, :]
            thought_vector = last_hidden_state.unsqueeze(1)
            
            current_embeds = thought_vector
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)
        
        # 3. Generate Answer (Token by Token)
        next_input_embeds = current_embeds
        
        all_log_probs = []
        all_values = []
        all_logits = []
        generated_answer_tokens = []

        for _ in range(self.max_answer_len):
            outputs = self.transformer(
                inputs_embeds=next_input_embeds,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
                use_cache=True
            )
            hidden_state = outputs.last_hidden_state
            
            # Actor: Output Logits for next token
            logits = self.lm_head(hidden_state)[:, -1, :]
            all_logits.append(logits)
            
            # Critic: Output Value estimate for this state
            value = self.critic_head(hidden_state)[:, -1, :]
            all_values.append(value)
            
            # Sample Action
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_answer_tokens.append(next_token)
            
            # Get Log Prob of the chosen action
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_prob = log_probs.gather(dim=1, index=next_token)
            all_log_probs.append(token_log_prob)
            
            # Update for next step
            past_key_values = outputs.past_key_values
            next_input_embeds = self.transformer.wte(next_token)
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)

        # Stack results
        answer_tensor = torch.stack(generated_answer_tokens, dim=1).squeeze(2)
        all_log_probs = torch.stack(all_log_probs, dim=1).squeeze(2)
        all_values = torch.stack(all_values, dim=1).squeeze(2)
        all_logits = torch.stack(all_logits, dim=1)
        
        return answer_tensor, all_log_probs, all_values, all_logits