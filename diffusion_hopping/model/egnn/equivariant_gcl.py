from typing import Optional

import torch
from torch import nn as nn
from torch_geometric.nn.conv import MessagePassing


class EquivariantGCL(MessagePassing):
    def __init__(
        self,
        hidden_features: int,
        edge_features: int,
        normalization_factor: float = 1.0,
        aggr: str = "add",
        use_tanh: bool = False,
        tanh_coords_range: float = 15.0,
        norm_constant: float = 1.0,
    ) -> None:
        super().__init__(aggr=aggr)

        last_layer = nn.Linear(hidden_features, 1, bias=False)
        nn.init.xavier_uniform_(last_layer.weight, gain=0.001)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_features + edge_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            last_layer,
        )

        self.use_tanh = use_tanh
        self.tanh_coords_range = tanh_coords_range
        self.normalization_factor = normalization_factor
        self.norm_constant = norm_constant

    def forward(
        self,
        x,
        pos,
        edge_index,
        mask,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr, mask=mask)

    def _get_pos_differences(self, pos_i, pos_j):
        pos_diff = pos_i - pos_j
        dist = torch.norm(pos_diff, dim=-1, keepdim=True)
        return pos_diff / (dist + self.norm_constant)

    def _calculate_translation_factor(self, concat):
        if self.use_tanh:
            translation = torch.tanh(self.mlp(concat)) * self.tanh_coords_range
        else:
            translation = self.mlp(concat)
        return translation

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr: Optional[torch.Tensor] = None):
        pos_diff = self._get_pos_differences(pos_i, pos_j)

        if edge_attr is None:
            concat = torch.cat([x_i, x_j], dim=-1)
        else:
            concat = torch.cat([x_i, x_j, edge_attr], dim=-1)

        translation = pos_diff * self._calculate_translation_factor(concat)
        return translation

    def update(self, aggr_out, pos, mask):
        if self.aggr == "add" or self.aggr == "sum":
            aggr_out = aggr_out / self.normalization_factor
        return pos + aggr_out * mask[..., None]

    def reset_parameters(self):
        self.mlp.reset_parameters()
