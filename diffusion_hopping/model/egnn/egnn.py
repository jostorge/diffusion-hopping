from typing import Optional

import torch
import torch.nn as nn

from diffusion_hopping.model.egnn.equivariant_block import EquivariantBlock
from diffusion_hopping.model.egnn.util import get_squared_distance


class EGNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int,
        hidden_features: int,
        num_layers: int,
        attention: bool = False,
        use_tanh: bool = True,
        tanh_coords_range: float = 15.0,
        inv_layers: int = 1,
        normalization_factor: float = 100.0,
        aggr: str = "add",
        norm_constant: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_in = nn.Linear(in_features, hidden_features)
        self.embedding_out = nn.Linear(hidden_features, out_features)

        self.layers = nn.ModuleList(
            [
                EquivariantBlock(
                    hidden_features,
                    edge_features + 1,
                    inv_layers,
                    attention=attention,
                    use_tanh=use_tanh,
                    tanh_coords_range=tanh_coords_range,
                    aggr=aggr,
                    normalization_factor=normalization_factor,
                    norm_constant=norm_constant,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x, pos, edge_index, mask, edge_attr: Optional[torch.Tensor] = None
    ):
        distances = get_squared_distance(pos, edge_index)
        if edge_attr is None:
            edge_attr = distances
        else:
            edge_attr = torch.cat([distances, edge_attr], dim=-1)

        x = self.embedding_in(x)
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, mask, edge_attr=edge_attr)
        x = self.embedding_out(x)
        return x, pos
