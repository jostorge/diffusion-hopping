from typing import Optional

import torch
from torch import nn as nn

from diffusion_hopping.model.egnn.equivariant_gcl import EquivariantGCL
from diffusion_hopping.model.egnn.gcl import GCL
from diffusion_hopping.model.egnn.util import get_squared_distance


class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_features: int,
        edge_features: int,
        num_layers: int,
        attention: bool = False,
        use_tanh: bool = False,
        tanh_coords_range: float = 15.0,
        aggr: str = "add",
        normalization_factor: float = 100.0,
        norm_constant: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GCL(
                    hidden_features,
                    hidden_features,
                    edge_features + 1,
                    hidden_features,
                    normalization_factor=normalization_factor,
                    aggr=aggr,
                    attention=attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.equivariant_gcl = EquivariantGCL(
            hidden_features,
            edge_features + 1,
            normalization_factor=normalization_factor,
            aggr=aggr,
            use_tanh=use_tanh,
            tanh_coords_range=tanh_coords_range,
            norm_constant=norm_constant,
        )

    def forward(
        self, x, pos, edge_index, mask, edge_attr: Optional[torch.Tensor] = None
    ):

        current_distances = get_squared_distance(pos, edge_index)
        if edge_attr is None:
            edge_attr = current_distances
        else:
            edge_attr = torch.cat([current_distances, edge_attr], dim=-1)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
        pos = self.equivariant_gcl(x, pos, edge_index, mask, edge_attr=edge_attr)
        return x, pos
