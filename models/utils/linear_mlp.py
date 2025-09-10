import torch
import torch.nn as nn

class LinearMLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        # print('x_in_linearMLP:', x.size())
        x = self.proj(x)
        return x