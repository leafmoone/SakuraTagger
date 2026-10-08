"""Group-specific queries over a single shared key/value projection."""
import torch
from torch import nn
from torch.nn import functional as F


class GroupAttention(nn.Module):
    def __init__(self, patch_dim, specs, hidden_dim=256, num_heads=4):
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError('Attention hidden_dim must be divisible by num_heads')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.names = tuple(spec.group_name for spec in specs)
        self.queries = nn.ParameterList(nn.Parameter(torch.empty(spec.query_count, hidden_dim)) for spec in specs)
        for query in self.queries:
            nn.init.normal_(query, std=0.02)
        self.key_value = nn.Linear(patch_dim, 2 * hidden_dim)

    def forward(self, patch_tokens, selected):
        active = [(name, query) for name, query in zip(self.names, self.queries) if name in selected]
        if not active:
            return {}
        batch, count, _ = patch_tokens.shape
        depth = self.hidden_dim // self.num_heads
        # Exactly one projection for all selected groups, regardless of group count.
        kv = self.key_value(patch_tokens).reshape(batch, count, 2, self.num_heads, depth)
        keys, values = kv.unbind(dim=2)
        keys, values = keys.transpose(1, 2), values.transpose(1, 2)
        queries = torch.cat([query for _, query in active], dim=0)
        q = queries.unsqueeze(0).expand(batch, -1, -1).reshape(batch, -1, self.num_heads, depth).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, keys, values, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, -1, self.hidden_dim)
        outputs = attended.split([query.shape[0] for _, query in active], dim=1)
        return {name: output.mean(dim=1) for (name, _), output in zip(active, outputs)}
