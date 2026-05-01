# gpt2.py
# Full GPT2Model assembly

import torch
import torch.nn as nn
from model.embeddings import GPT2Embeddings
from model.block import TransformerBlock


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = GPT2Embeddings(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.embeddings.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
