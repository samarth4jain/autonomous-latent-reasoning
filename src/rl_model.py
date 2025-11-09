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

        # 2. Generate N thoughts (we don't need their log-probs)
        for _ in range(self.n_thoughts):
            outputs = self.transformer(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask
            )
            
            last_hidden_state = outputs.last_hidden_state[:, -1, :]
            thought_vector = last_hidden_state.unsqueeze(1)
            
            current_embeds = torch.cat([current_embeds, thought_vector], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)
        
        # 3. Generate the answer, one token at a time
        
        # Get the final hidden state after all thoughts
        last_hidden_state = current_embeds[:, -1, :].unsqueeze(1)
        
        total_log_prob = torch.zeros(batch_size, device=device)
        generated_answer_tokens = []

        # Start with the last hidden state as the first input for the decoder
        current_decoder_embeds = last_hidden_state

        for _ in range(self.max_answer_len):
            # Get logits from the last token's hidden state
            outputs = self.transformer(inputs_embeds=current_decoder_embeds, attention_mask=current_attention_mask)
            next_token_logits = self.lm_head(outputs.last_hidden_state)[:, -1, :]
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            
            # Get the log-probability of the token we just sampled
            # We use log_softmax for numerical stability
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            
            # Gather the log-prob of the chosen token
            # (batch_size, 1) -> (batch_size)
            chosen_token_log_prob = log_probs.gather(dim=1, index=next_token).squeeze(1)
            
            # Add to the total log-prob for this sequence
            total_log_prob = total_log_prob + chosen_token_log_prob
            
            # Append the chosen token to our answer
            generated_answer_tokens.append(next_token)
            
            # Get the embedding of the new token and use it as the next input
            current_decoder_embeds = self.transformer.wte(next_token)
            # Update the attention mask
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)

        # Stack all generated answer tokens into a tensor
        # (batch_size, max_answer_len)
        answer_tensor = torch.stack(generated_answer_tokens, dim=1).squeeze(2)
        
        # Return the generated answer and the total log-prob of that answer
        return answer_tensor, total_log_prob