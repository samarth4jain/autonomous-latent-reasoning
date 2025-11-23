import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class ContinuousThoughtModel(GPT2LMHeadModel):
    def __init__(self, config, n_thoughts=6):
        super().__init__(config)
        self.n_thoughts = n_thoughts

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        **kwargs
    ):
        # 1. Get initial embeddings
        inputs_embeds = self.transformer.wte(input_ids)
        current_embeds = inputs_embeds
        current_attention_mask = attention_mask

        # 2. LOGIC: Only generate thoughts if this is the *first* pass
        # If past_key_values is None, we are processing the prompt -> Generate Thoughts
        # If past_key_values exists, we are generating tokens one-by-one -> Skip Thoughts
        if past_key_values is None:
            ones_mask = torch.ones(
                current_embeds.shape[0], 1,
                dtype=torch.long, device=current_embeds.device
            )
            
            for _ in range(self.n_thoughts):
                outputs = self.transformer(
                    inputs_embeds=current_embeds,
                    attention_mask=current_attention_mask,
                    use_cache=True # Ensure we build up state
                )
                
                last_hidden_state = outputs.last_hidden_state[:, -1, :]
                thought_vector = last_hidden_state.unsqueeze(1)
                
                current_embeds = torch.cat([current_embeds, thought_vector], dim=1)
                if current_attention_mask is not None:
                    current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)

        # 3. Standard Transformer Forward Pass
        # This handles both the initial thought generation AND the subsequent answer generation
        final_transformer_outputs = self.transformer(
            inputs_embeds=current_embeds,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        hidden_states = final_transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        # 4. Calculate Loss (Training Mode Only)
        loss = None
        if labels is not None:
            # During training, we don't use generate(), so past_key_values is None.
            # The 'logits' here contain [Question + Thoughts + Answer].
            # We must align this with the 'labels'.
            
            # We only care about predicting the answer tokens.
            # The logic in train_baseline / train_lpo handles the alignment by
            # slicing logits or masking labels.
            
            loss_fct = nn.CrossEntropyLoss()
            
            # Align labels (answer) with the thought logits
            # Logits: (batch_size, N_THOUGHTS, vocab_size) if focusing on thought output
            # But here we have full sequence logits. 
            
            # Assuming the standard LPO training loop which slices logits manually:
            thought_logits = lm_logits[:, -self.n_thoughts:, :]
            
            # Safe handling for varying label lengths
            num_tokens_to_compare = min(self.n_thoughts, labels.shape[1])
            
            shift_logits = thought_logits[:, :num_tokens_to_compare, :].contiguous()
            shift_labels = labels[:, :num_tokens_to_compare].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {
            "loss": loss, 
            "logits": lm_logits,
            "past_key_values": final_transformer_outputs.past_key_values
        }