import torch.nn as nn
from typing import Sequence


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, depth=1, act_last=True, act=nn.ReLU, bias=True, dropout=0):
        super().__init__()
        layers = []
        for i in range(depth):
            d1 = in_dim if i == 0 else hidden_dim
            d2 = out_dim if i == depth - 1 else hidden_dim
            layers.append(nn.Linear(d1, d2, bias=bias))
            if i != depth - 1 or act_last:
                layers.append(act())
            if dropout > 0 and i != depth - 1:
                layers.append(nn.Dropout(p=dropout))
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)
