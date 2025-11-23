import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class ContinuousThoughtModel(GPT2LMHeadModel):
    def __init__(self, config, n_thoughts=6):
        super().__init__(config)
        self.n_thoughts = n_thoughts

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # 1. Handle input embeddings
        # If input_ids is provided, get embeddings. Otherwise use provided inputs_embeds
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        
        current_embeds = inputs_embeds
        current_attention_mask = attention_mask

        # 2. LOGIC: Only generate thoughts if this is the *first* pass (Prompt processing)
        # If past_key_values is None, we are processing the prompt -> Generate Thoughts
        # If past_key_values exists, we are generating answer tokens -> Skip Thoughts
        if past_key_values is None and current_embeds is not None:
            batch_size = current_embeds.shape[0]
            device = current_embeds.device
            ones_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            for _ in range(self.n_thoughts):
                outputs = self.transformer(
                    inputs_embeds=current_embeds,
                    attention_mask=current_attention_mask,
                    use_cache=True # Ensure we build up state
                )
                
                last_hidden_state = outputs.last_hidden_state[:, -1, :]
                thought_vector = last_hidden_state.unsqueeze(1)
                
                current_embeds = torch.cat([current_embeds, thought_vector], dim=1)
                
                # Extend attention mask for the new thought token
                if current_attention_mask is not None:
                    current_attention_mask = torch.cat([current_attention_mask, ones_mask], dim=1)

        # 3. Standard Transformer Forward Pass
        # This handles both the initial thought generation AND subsequent answer generation
        transformer_outputs = self.transformer(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            attention_mask=current_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        # 4. Calculate Loss (Training Mode Only)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # We assume the training loop handles the masking of thoughts vs answer
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 5. Return Standard Output Object (Fixes the AttributeError)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )