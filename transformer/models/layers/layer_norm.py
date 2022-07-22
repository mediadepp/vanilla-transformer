from calendar import c
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) 
        self.beta = nn.Parameter(torch.zeros(d_model)) 
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = out * self.gamma + self.beta
        return out
        