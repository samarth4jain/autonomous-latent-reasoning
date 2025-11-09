import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class ReinforceModel(GPT2LMHeadModel):
    def __init__(self, config, n_thoughts, max_answer_len):
        super().__init__(config)
        self.n_thoughts = n_thoughts
        self.max_answer_len = max_answer_len

    def forward(self, input_ids, attention_mask):
        """
        This is the main 'generate' function for our REINFORCE agent.
        It generates a sequence of thoughts, then an answer, and 
        calculates the log-probability of the answer it chose.
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
                past_key_values=past_key_values
            )
            
            # Get the new KV cache
            past_key_values = outputs.past_key_values
            
            # Get the last hidden state
            last_hidden_state = outputs.last_hidden_state[:, -1, :]
            thought_vector = last_hidden_state.unsqueeze(1)
            
            # The next input is the thought vector
            current_embeds = thought_vector
            
            # Update the attention mask
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)
        
        # 3. Generate the answer, one token at a time
        
        # The 'current_embeds' is the last thought vector. We use it as the
        # first input to the answer-generation loop.
        next_input_embeds = current_embeds
        
        total_log_prob = torch.zeros(batch_size, device=device)
        generated_answer_tokens = []

        for _ in range(self.max_answer_len):
            # Pass the *past_key_values* and *only the new token embedding*
            outputs = self.transformer(
                inputs_embeds=next_input_embeds,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask 
            )
            
            # Get the logits for the *new* token
            next_token_logits = self.lm_head(outputs.last_hidden_state)[:, -1, :]
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            
            # Get the log-probability of the token we just sampled
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            
            # Gather the log-prob of the chosen token
            chosen_token_log_prob = log_probs.gather(dim=1, index=next_token).squeeze(1)
            
            # Add to the total log-prob for this sequence
            # We add it only if it's not a padding token
            # This is a bit complex, let's simplify and just sum all
            total_log_prob = total_log_prob + chosen_token_log_prob
            
            # Append the chosen token to our answer
            generated_answer_tokens.append(next_token)
            
            # --- Prepare for next iteration ---
            # The *new* past_key_values are the output from this step
            past_key_values = outputs.past_key_values
            # The *new* input is the embedding of the token we just sampled
            next_input_embeds = self.transformer.wte(next_token)
            # The *new* attention mask is just one token longer
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)

        # Stack all generated answer tokens into a tensor
        # (batch_size, max_answer_len)
        answer_tensor = torch.stack(generated_answer_tokens, dim=1).squeeze(2)
        
        # Return the generated answer and the total log-prob of that answer
        return answer_tensor, total_log_prob