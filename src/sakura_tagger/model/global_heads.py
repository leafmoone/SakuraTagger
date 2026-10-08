from torch import nn


class GlobalHeads(nn.Module):
    def __init__(self, global_dim, specs):
        super().__init__()
        self.names = tuple(spec.group_name for spec in specs)
        self.heads = nn.ModuleList(nn.Linear(global_dim, spec.num_classes) for spec in specs)

    def forward(self, global_features, selected):
        return {name: head(global_features) for name, head in zip(self.names, self.heads) if name in selected}
