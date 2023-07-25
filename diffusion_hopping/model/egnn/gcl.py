import torch
from torch import nn as nn
from torch_geometric.nn.conv import MessagePassing


class GCL(MessagePassing):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int,
        hidden_features: int,
        normalization_factor: float = 1.0,
        aggr: str = "add",
        attention: bool = False,
    ):
        super().__init__(aggr=aggr)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_features + edge_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

        self.attention = attention
        if self.attention:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_features, 1),
                nn.Sigmoid(),
            )

        self.normalization_factor = normalization_factor

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # concat x_i, x_j, edge_attr
        out = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out = self.edge_mlp(out)
        if self.attention:
            att = self.attention_net(out)
            return out * att
        else:
            return out

    def update(self, aggr_out, x):
        if self.aggr == "add" or self.aggr == "sum":
            aggr_out = aggr_out / self.normalization_factor
        out = torch.cat([x, aggr_out], dim=-1)
        out = x + self.node_mlp(out)
        return out

    def reset_parameters(self):
        self.node_mlp.reset_parameters()
        self.edge_mlp.reset_parameters()
        if self.attention:
            self.attention_net.reset_parameters()
