"""Query-preserving attention for the single General classification module."""
import torch
from torch import nn
from torch.nn import functional as F


class GroupAttention(nn.Module):
    def __init__(self, patch_dim, query_count=16, hidden_dim=256, num_heads=4):
        super().__init__()
        if min(query_count, hidden_dim, num_heads) < 1 or hidden_dim % num_heads:
            raise ValueError('Positive attention dimensions must divide evenly into heads')
        self.hidden_dim, self.num_heads = hidden_dim, num_heads
        self.queries = nn.Parameter(torch.empty(query_count, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.key_value = nn.Linear(patch_dim, 2 * hidden_dim)

    def forward(self, patch_tokens):
        batch, count, _ = patch_tokens.shape
        depth = self.hidden_dim // self.num_heads
        kv = self.key_value(patch_tokens).reshape(batch, count, 2, self.num_heads, depth)
        keys, values = (x.transpose(1, 2) for x in kv.unbind(dim=2))
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        queries = queries.reshape(batch, -1, self.num_heads, depth).transpose(1, 2)
        attended = F.scaled_dot_product_attention(queries, keys, values, dropout_p=0.0)
        return attended.transpose(1, 2).reshape(batch, -1, self.hidden_dim)
