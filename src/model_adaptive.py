import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class GPT2AdaptiveLatentReasoning(GPT2LMHeadModel):
    def __init__(self, config, max_thoughts=15):
        super().__init__(config)
        self.max_thoughts = max_thoughts
        self.halt_head = nn.Linear(config.n_embd, 1)
        self.current_thoughts_used = 0 

    def forward(self, input_ids=None, attention_mask=None, labels=None, past_key_values=None, generate_thoughts=True, **kwargs):
        ponder_cost = 0.0

        if past_key_values is not None:
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                kwargs["position_ids"] = kwargs["position_ids"] + self.current_thoughts_used
            if attention_mask is not None:
                device = attention_mask.device
                padding = torch.ones((attention_mask.shape[0], self.current_thoughts_used), dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([attention_mask, padding], dim=1)

        if generate_thoughts and past_key_values is None and input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            inputs_embeds = self.transformer.wte(input_ids)
            current_attention_mask = attention_mask
            
            thoughts_used = 0
            for step in range(self.max_thoughts):
                outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=current_attention_mask)
                thought_vector = outputs.last_hidden_state[:, -1, :].unsqueeze(1)
                
                halt_logit = self.halt_head(thought_vector.squeeze(1))
                halt_prob = torch.sigmoid(halt_logit)
                
                ponder_cost = ponder_cost + halt_prob.mean() 
                thoughts_used += 1
                
                inputs_embeds = torch.cat([inputs_embeds, thought_vector], dim=1)
                if current_attention_mask is not None:
                    ones = torch.ones((batch_size, 1), device=device)
                    current_attention_mask = torch.cat([current_attention_mask, ones], dim=1)

                # THE FIX: Take the mean across all beams!
                if not self.training and halt_prob.mean().item() > 0.5:
                    break 
            
            self.current_thoughts_used = thoughts_used

            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                extra_positions = kwargs["position_ids"][:, -1:] + torch.arange(1, thoughts_used + 1, device=device).unsqueeze(0)
                kwargs["position_ids"] = torch.cat([kwargs["position_ids"], extra_positions], dim=1)

            transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=current_attention_mask, **kwargs)
        else:
            transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

        lm_logits = self.lm_head(transformer_outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = ce_loss + (0.0001 * ponder_cost) 

        return CausalLMOutputWithCrossAttentions(
            loss=loss, logits=lm_logits, past_key_values=transformer_outputs.past_key_values
        )
