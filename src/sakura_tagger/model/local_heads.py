import torch
from torch import nn

from .group_attention import GroupAttention


class LocalHeads(nn.Module):
    def __init__(self, global_dim, patch_dim, specs, hidden_dim=256, num_heads=4):
        super().__init__()
        self.names = tuple(spec.group_name for spec in specs)
        self.attention = GroupAttention(patch_dim, specs, hidden_dim, num_heads)
        self.heads = nn.ModuleList(nn.Linear(global_dim + hidden_dim, spec.num_classes) for spec in specs)

    def forward(self, global_features, patch_tokens, selected):
        local = self.attention(patch_tokens, selected)
        return {name: head(torch.cat((global_features, local[name]), dim=-1))
                for name, head in zip(self.names, self.heads) if name in selected}
