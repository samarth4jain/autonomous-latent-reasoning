import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class ContinuousThoughtModel(GPT2LMHeadModel):
    def __init__(self, config, n_thoughts=6):
        super().__init__(config)
        self.n_thoughts = n_thoughts

    def forward(self, input_ids=None, attention_mask=None, labels=None, past_key_values=None, **kwargs):
        
        # --- POSITIONAL ID FIX ---
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            if past_key_values is None:
                device = kwargs["position_ids"].device
                extra_positions = kwargs["position_ids"][:, -1:] + torch.arange(1, self.n_thoughts + 1, device=device).unsqueeze(0)
                kwargs["position_ids"] = torch.cat([kwargs["position_ids"], extra_positions], dim=1)
            else:
                kwargs["position_ids"] = kwargs["position_ids"] + self.n_thoughts

        # --- ATTENTION MASK FIX ---
        if past_key_values is not None and attention_mask is not None:
            # Hugging Face forgets the thoughts we added, so we pad the mask to match the cache
            device = attention_mask.device
            padding = torch.ones((attention_mask.shape[0], self.n_thoughts), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, padding], dim=1)

        # 1. First Pass: Generate Thoughts
        if past_key_values is None and input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            inputs_embeds = self.transformer.wte(input_ids)
            current_attention_mask = attention_mask
            
            for _ in range(self.n_thoughts):
                outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=current_attention_mask)
                thought_vector = outputs.last_hidden_state[:, -1, :].unsqueeze(1)
                
                inputs_embeds = torch.cat([inputs_embeds, thought_vector], dim=1)
                if current_attention_mask is not None:
                    ones = torch.ones((batch_size, 1), device=device)
                    current_attention_mask = torch.cat([current_attention_mask, ones], dim=1)
            
            transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=current_attention_mask, **kwargs)
        else:
            # 2. Subsequent Passes: Standard KV-cached generation
            transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

        lm_logits = self.lm_head(transformer_outputs.last_hidden_state)
        
        # 3. Calculate Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss, logits=lm_logits, past_key_values=transformer_outputs.past_key_values
        )
