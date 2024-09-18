"""
Implement bigram language model

For an in-depth explanation, refer to the makemore series on youtube by Andrej Karpathy
"""

import torch
import torch.nn as nn # neural network module
import torch.nn.functional as F # functional interface

torch.manual_seed(1337) # set seed for reproducibility

class BigramLanguageModel(nn.Module):
    """
    Bigram language model is a simple model that predicts the next token in a sequence given the previous token
    """
    def __init__(self, vocab_size):
        super().__init__() # inherit from nn.Module 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # word embedding table

    def forward(self, idx, targets=None):
        """
        Forward pass of the model. We pass in the input tokens and the targets and the model will output the loss

        Size of idx is (B, T) where B is the batch size and T is the sequence length
        Size of targets, if exists, is also (B, T)
        """

        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # reshape so it is compatible with the cross entropy loss function 
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)


        return logits, loss
    
    torch.no_grad() # disable gradient computation
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens based on the given context.

        Args:
            idx (torch.Tensor): The input context tensor of shape (B, T).
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            torch.Tensor: The generated sequence of tokens.
        """
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, _ = self.forward(idx)
            # logits shape is (B, T, C)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # pick the highest prob token
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

