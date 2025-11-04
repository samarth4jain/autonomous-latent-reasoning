import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class ContinuousThoughtModel(GPT2LMHeadModel):
    def __init__(self, config, n_thoughts):
        super().__init__(config)
        self.n_thoughts = n_thoughts

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        # 1. Get embeddings for the question
        inputs_embeds = self.transformer.wte(input_ids)
        current_embeds = inputs_embeds
        
        # 2. Create the attention mask for the question
        current_attention_mask = attention_mask

        # 3. Create a mask of '1s' to append for each thought
        ones_mask = torch.ones(
            current_embeds.shape[0], 1,
            dtype=torch.long, device=current_embeds.device
        )

        # 4. Loop to generate N thoughts
        # This implements the "w/o curriculum" baseline [cite: 203-205]
        for _ in range(self.n_thoughts):
            outputs = self.transformer(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask
            )
            
            last_hidden_state = outputs.last_hidden_state[:, -1, :]
            thought_vector = last_hidden_state.unsqueeze(1)
            
            current_embeds = torch.cat([current_embeds, thought_vector], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)
        
        # 5. Get the final logits after all thoughts
        final_transformer_outputs = self.transformer(
            inputs_embeds=current_embeds,
            attention_mask=current_attention_mask
        )
        hidden_states = final_transformer_outputs.last_hidden_state
        
        # 6. We only care about the logits from the *last N_THOUGHTS tokens*
        # These will be used to predict the answer, which is in 'labels'
        lm_logits = self.lm_head(hidden_states)
        
        # Get the logits for the 'thought' positions
        thought_logits = lm_logits[:, -self.n_thoughts:, :]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # Align labels (answer) with the thought logits
            # Logits: (batch_size, N_THOUGHTS, vocab_size)
            # Labels: (batch_size, MAX_ANSWER_LEN)
            
            num_tokens_to_compare = min(self.n_thoughts, labels.shape[1])
            
            shift_logits = thought_logits[:, :num_tokens_to_compare, :].contiguous()
            shift_labels = labels[:, :num_tokens_to_compare].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": thought_logits}