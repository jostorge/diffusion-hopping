from abc import ABC
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing

from diffusion_hopping.model.gvp.dropout import GVPDropout
from diffusion_hopping.model.gvp.gvp import GVP, s_V
from diffusion_hopping.model.gvp.layer_norm import GVPLayerNorm


class GVPMessagePassing(MessagePassing, ABC):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        hidden_dims: Optional[Tuple[int, int]] = None,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        attention: bool = True,
        aggr: str = "add",
        normalization_factor: float = 1.0,
    ):
        super().__init__(aggr)
        if hidden_dims is None:
            hidden_dims = out_dims

        in_scalar, in_vector = in_dims
        hidden_scalar, hidden_vector = hidden_dims

        edge_scalar, edge_vector = edge_dims

        self.out_scalar, self.out_vector = out_dims
        self.in_vector = in_vector
        self.hidden_scalar = hidden_scalar
        self.hidden_vector = hidden_vector
        self.normalization_factor = normalization_factor

        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)
        self.edge_gvps = nn.Sequential(
            GVP_(
                (2 * in_scalar + edge_scalar, 2 * in_vector + edge_vector),
                hidden_dims,
            ),
            GVP_(hidden_dims, hidden_dims),
            GVP_(hidden_dims, out_dims, activations=(None, None)),
        )

        self.attention = attention
        if attention:
            self.attention_gvp = GVP_(
                out_dims,
                (1, 0),
                activations=(torch.sigmoid, None),
            )

    def forward(self, x: s_V, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> s_V:
        s, V = x
        v_dim = V.shape[-1]
        V = torch.flatten(V, start_dim=-2, end_dim=-1)
        return self.propagate(edge_index, s=s, V=V, edge_attr=edge_attr, v_dim=v_dim)

    def message(self, s_i, s_j, V_i, V_j, edge_attr, v_dim):
        V_i = V_i.view(*V_i.shape[:-1], self.in_vector, v_dim)
        V_j = V_j.view(*V_j.shape[:-1], self.in_vector, v_dim)
        edge_scalar, edge_vector = edge_attr

        s = torch.cat([s_i, s_j, edge_scalar], dim=-1)
        V = torch.cat([V_i, V_j, edge_vector], dim=-2)
        s, V = self.edge_gvps((s, V))

        if self.attention:
            att = self.attention_gvp((s, V))
            s, V = att * s, att[..., None] * V
        return self._combine(s, V)

    def update(self, aggr_out: torch.Tensor) -> s_V:
        s_aggr, V_aggr = self._split(aggr_out, self.out_scalar, self.out_vector)
        if self.aggr == "add" or self.aggr == "sum":
            s_aggr = s_aggr / self.normalization_factor
            V_aggr = V_aggr / self.normalization_factor
        return s_aggr, V_aggr

    @staticmethod
    def _combine(s, V) -> torch.Tensor:
        V = torch.flatten(V, start_dim=-2, end_dim=-1)
        return torch.cat([s, V], dim=-1)

    @staticmethod
    def _split(s_V: torch.Tensor, scalar: int, vector: int) -> s_V:
        s = s_V[..., :scalar]
        V = s_V[..., scalar:]
        V = V.view(*V.shape[:-1], vector, -1)
        return s, V

    def reset_parameters(self):
        for gvp in self.edge_gvps:
            gvp.reset_parameters()
        if self.attention:
            self.attention_gvp.reset_parameters()


class GVPConvLayer(GVPMessagePassing, ABC):
    def __init__(
        self,
        node_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        drop_rate: float = 0.0,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        residual: bool = True,
        attention: bool = True,
        aggr: str = "add",
        normalization_factor: float = 1.0,
    ):
        super().__init__(
            node_dims,
            node_dims,
            edge_dims,
            hidden_dims=node_dims,
            activations=activations,
            vector_gate=vector_gate,
            attention=attention,
            aggr=aggr,
            normalization_factor=normalization_factor,
        )
        self.residual = residual
        self.drop_rate = drop_rate
        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([GVPLayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([GVPDropout(drop_rate) for _ in range(2)])

        self.ff_func = nn.Sequential(
            GVP_(node_dims, node_dims),
            GVP_(node_dims, node_dims, activations=(None, None)),
        )
        self.residual = residual

    def forward(
        self,
        x: Union[s_V, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> s_V:

        s, V = super().forward(x, edge_index, edge_attr)
        if self.residual:
            s, V = self.dropout[0]((s, V))
            s, V = x[0] + s, x[1] + V
            s, V = self.norm[0]((s, V))

        x = (s, V)
        s, V = self.ff_func(x)

        if self.residual:
            s, V = self.dropout[1]((s, V))
            s, V = s + x[0], V + x[1]
            s, V = self.norm[1]((s, V))

        return s, V
