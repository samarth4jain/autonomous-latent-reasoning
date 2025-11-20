import torch
import torch.nn as nn

class RNDModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, output_size=512):
        super(RNDModel, self).__init__()
        
        # 1. Target Network (Fixed, Random)
        # This represents the "unknown". It never changes.
        self.target = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        )
        
        # 2. Predictor Network (Trainable)
        # Tries to predict the Target. High error = High Curiosity.
        self.predictor = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )

        # Freeze the target network permanently
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, token_ids):
        # Get features
        target_features = self.target(token_ids)       
        predictor_features = self.predictor(token_ids) 
        
        # Calculate Prediction Error (MSE)
        # The agent is "curious" about sequences where the predictor fails
        error = (predictor_features - target_features).pow(2)
        
        # Loss to train the predictor (it wants to minimize surprise)
        rnd_loss = torch.mean(error)
        
        # Reward for the agent (it wants to maximize surprise/novelty)
        # We mean over the embedding dim (2), but keep batch (0) and seq (1)
        # Then mean over sequence to get reward per batch item
        # We detach because the agent shouldn't optimize the RND module directly
        with torch.no_grad():
            intrinsic_reward = torch.mean(error, dim=2).mean(dim=1)
            
        return rnd_loss, intrinsic_reward