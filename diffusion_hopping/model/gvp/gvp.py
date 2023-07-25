import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

s_V = Tuple[torch.Tensor, torch.Tensor]


# Relevant papers:
# Learning from Protein Structure with Geometric Vector Perceptrons,
# Equivariant Graph Neural Networks for 3D Macromolecular Structure,
class GVP(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        in_scalar, in_vector = in_dims
        out_scalar, out_vector = out_dims
        self.sigma, self.sigma_plus = activations

        if self.sigma is None:
            self.sigma = nn.Identity()
        if self.sigma_plus is None:
            self.sigma_plus = nn.Identity()

        self.h = max(in_vector, out_vector)
        self.W_h = nn.Parameter(torch.empty((self.h, in_vector)))
        self.W_mu = nn.Parameter(torch.empty((out_vector, self.h)))

        self.W_m = nn.Linear(self.h + in_scalar, out_scalar)
        self.v = in_vector
        self.mu = out_vector
        self.n = in_scalar
        self.m = out_scalar
        self.vector_gate = vector_gate

        if vector_gate:
            self.sigma_g = nn.Sigmoid()
            self.W_g = nn.Linear(out_scalar, out_vector)

        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W_h, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        self.W_m.reset_parameters()
        if self.vector_gate:
            self.W_g.reset_parameters()

    def forward(self, x: Union[torch.Tensor, s_V]) -> Union[torch.Tensor, s_V]:
        """Geometric vector perceptron"""
        s, V = (
            x if self.v > 0 else (x, torch.empty((x.shape[0], 0, 3), device=x.device))
        )

        assert (
            s.shape[-1] == self.n
        ), f"{s.shape[-1]} != {self.n} Scalar dimension mismatch"
        assert (
            V.shape[-2] == self.v
        ), f" {V.shape[-2]} != {self.v} Vector dimension mismatch"
        assert V.shape[0] == s.shape[0], "Batch size mismatch"

        V_h = self.W_h @ V
        V_mu = self.W_mu @ V_h
        s_h = torch.clip(torch.norm(V_h, dim=-1), min=self.eps)
        s_hn = torch.cat([s, s_h], dim=-1)
        s_m = self.W_m(s_hn)
        s_dash = self.sigma(s_m)
        if self.vector_gate:
            V_dash = self.sigma_g(self.W_g(self.sigma_plus(s_m)))[..., None] * V_mu
        else:
            v_mu = torch.clip(torch.norm(V_mu, dim=-1, keepdim=True), min=self.eps)
            V_dash = self.sigma_plus(v_mu) * V_mu
        return (s_dash, V_dash) if self.mu > 0 else s_dash
