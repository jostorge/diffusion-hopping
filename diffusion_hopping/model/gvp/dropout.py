from typing import Union

import torch
from torch import nn as nn

from diffusion_hopping.model.gvp.gvp import s_V


class GVPDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.dropout_features = nn.Dropout(p)
        self.dropout_vector = nn.Dropout1d(p)

    def forward(self, x: Union[torch.Tensor, s_V]) -> Union[torch.Tensor, s_V]:
        if isinstance(x, torch.Tensor):
            return self.dropout_features(x)

        s, V = x
        s = self.dropout_features(s)
        V = self.dropout_vector(V)
        return s, V
