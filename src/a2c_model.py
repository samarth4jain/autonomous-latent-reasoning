import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class A2CModel(GPT2LMHeadModel):
    def __init__(self, config, n_thoughts, max_answer_len):
        super().__init__(config)
        self.n_thoughts = n_thoughts
        self.max_answer_len = max_answer_len
        
        # --- NEW ---
        # Add a "Critic" head: a simple linear layer that maps the
        # hidden state to a single scalar "value" (score).
        self.critic_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        This is the main 'generate' function for our A2C agent.
        It generates an answer and, at each step, outputs:
        1. The log-probability of the token it chose (for the Actor)
        2. The "value" of that state (for the Critic)
        """
        
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 1. Get embeddings for the question
        inputs_embeds = self.transformer.wte(input_ids)
        current_embeds = inputs_embeds
        current_attention_mask = attention_mask

        ones_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        past_key_values = None

        # 2. Generate N thoughts (we don't need their log-probs)
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
        
        # 3. Generate the answer, one token at a time
        next_input_embeds = current_embeds
        
        all_log_probs = []
        all_values = []
        generated_answer_tokens = []

        for _ in range(self.max_answer_len):
            outputs = self.transformer(
                inputs_embeds=next_input_embeds,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
                use_cache=True
            )
            
            # Get the hidden state for the new token
            hidden_state = outputs.last_hidden_state
            
            # --- ACTOR ---
            next_token_logits = self.lm_head(hidden_state)[:, -1, :]
            log_probs_dist = torch.log_softmax(next_token_logits, dim=-1)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # --- CRITIC ---
            # Get the Critic's "value" prediction for this state
            value = self.critic_head(hidden_state)[:, -1, :] # (batch_size, 1)
            all_values.append(value)

            # Get the log-prob of the token we just sampled
            chosen_token_log_prob = log_probs_dist.gather(dim=1, index=next_token)
            all_log_probs.append(chosen_token_log_prob)
            
            # Append the chosen token to our answer
            generated_answer_tokens.append(next_token)
            
            # --- Prepare for next iteration ---
            past_key_values = outputs.past_key_values
            next_input_embeds = self.transformer.wte(next_token)
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)

        # Stack all generated answer tokens, log-probs, and values
        answer_tensor = torch.stack(generated_answer_tokens, dim=1).squeeze(2)
        all_log_probs = torch.stack(all_log_probs, dim=1).squeeze(2)
        all_values = torch.stack(all_values, dim=1).squeeze(2)
        
        return answer_tensor, all_log_probs, all_values