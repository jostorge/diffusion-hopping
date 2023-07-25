from typing import Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from diffusion_hopping.model.gvp.conv_layer import GVPConvLayer
from diffusion_hopping.model.gvp.gvp import GVP, s_V
from diffusion_hopping.model.gvp.layer_norm import GVPLayerNorm


class GVPNetwork(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        hidden_dims: Tuple[int, int],
        num_layers: int,
        attention: bool = False,
        normalization_factor: float = 100.0,
        aggr: str = "add",
        activations=(F.silu, None),
        vector_gate: bool = True,
        eps=1e-4,
    ) -> None:
        super().__init__()
        edge_dims = (1, 1)

        self.eps = eps
        self.embedding_in = nn.Sequential(
            GVPLayerNorm(in_dims),
            GVP(
                in_dims, hidden_dims, activations=(None, None), vector_gate=vector_gate
            ),
        )
        self.embedding_out = nn.Sequential(
            GVPLayerNorm(hidden_dims),
            GVP(
                hidden_dims, out_dims, activations=activations, vector_gate=vector_gate
            ),
        )
        self.edge_embedding = nn.Sequential(
            GVPLayerNorm(edge_dims),
            GVP(
                edge_dims,
                (hidden_dims[0], 1),
                activations=(None, None),
                vector_gate=vector_gate,
            ),
        )

        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    hidden_dims,
                    (hidden_dims[0], 1),
                    activations=activations,
                    vector_gate=vector_gate,
                    residual=True,
                    attention=attention,
                    aggr=aggr,
                    normalization_factor=normalization_factor,
                )
                for _ in range(num_layers)
            ]
        )

    def get_edge_attr(self, edge_index, pos) -> s_V:
        V = pos[edge_index[0]] - pos[edge_index[1]]  # [n_edges, 3]
        s = torch.linalg.norm(V, dim=-1, keepdim=True)  # [n_edges, 1]
        V = (V / torch.clip(s, min=self.eps))[..., None, :]  # [n_edges, 1, 3]
        return s, V

    def forward(self, x, pos, edge_index) -> s_V:
        edge_attr = self.get_edge_attr(edge_index, pos)
        edge_attr = self.edge_embedding(edge_attr)

        x = self.embedding_in(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return self.embedding_out(x)
