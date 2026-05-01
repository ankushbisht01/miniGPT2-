# mlp.py
# Feed-forward block

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
