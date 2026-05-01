# embeddings.py
# Token + positional embeddings

import torch
import torch.nn as nn


class GPT2Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.dropout(self.token_emb(input_ids) + self.pos_emb(positions))
