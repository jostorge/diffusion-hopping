from abc import ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F


class Schedule(nn.Module, metaclass=ABCMeta):
    def __init__(self, beta) -> None:
        super().__init__()
        buffers = {}
        buffers["beta"] = beta
        buffers["alpha"] = 1.0 - beta
        buffers["alpha_bar"] = torch.cumprod(buffers["alpha"], axis=0)
        buffers["sqrt_alpha"] = torch.sqrt(buffers["alpha"])
        buffers["sqrt_recip_alpha"] = torch.sqrt(1.0 / buffers["alpha"])
        buffers["sqrt_alpha_bar"] = torch.sqrt(buffers["alpha_bar"])
        buffers["sqrt_one_minus_alpha_bar"] = torch.sqrt(1.0 - buffers["alpha_bar"])

        alpha_bar_prev = F.pad(buffers["alpha_bar"][:-1], (1, 0), value=1.0)
        # either sigma_t = beta_t or sigma_t = beta_bar_t = (1-alpha_bar_(t-1))/(1-alpha_bar_t)
        # see https://arxiv.org/pdf/2006.11239.pdf
        buffers["posterior_variance"] = (
            buffers["beta"] * (1.0 - alpha_bar_prev) / (1.0 - buffers["alpha_bar"])
        )

        for k, v in buffers.items():
            self.register_buffer(k, v, persistent=False)


class LinearBetaSchedule(Schedule):
    def __init__(self, T) -> None:
        beta_start = 0.0001
        beta_end = 0.02

        beta = torch.linspace(beta_start, beta_end, T)
        super().__init__(beta)


class CosineBetaSchedule(Schedule):
    def __init__(self, T, s=0.008, beta_min=0, beta_max=0.999):
        """cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""

        x = torch.linspace(0, 1, T)
        f_t = torch.cos((x + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        alpha_bar_dash = torch.cat([torch.tensor([1.0]), alpha_bar])
        beta = 1 - (alpha_bar_dash[1:] / alpha_bar_dash[:-1])
        beta = torch.clip(beta, beta_min, beta_max)
        super().__init__(beta)


class PolynomialBetaSchedule(Schedule):
    def __init__(self, T, power=2.0, s=1e-4) -> None:
        x = torch.linspace(0, 1, T)
        alpha_bar = (1 - torch.pow(x, power)) ** 2
        alpha_bar = clip_noise_schedule(alpha_bar, clip_value=0.001)

        precision = 1 - 2 * s
        alpha_bar = precision * alpha_bar + s

        alpha_bar_dash = torch.cat([torch.tensor([1.0]), alpha_bar])
        beta = 1 - (alpha_bar_dash[1:] / alpha_bar_dash[:-1])
        super().__init__(beta)


def clip_noise_schedule(alpha_bar, clip_value=0.001):
    alpha_bar_dash = torch.cat([torch.tensor([1.0]), alpha_bar])

    alpha = alpha_bar_dash[1:] / alpha_bar_dash[:-1]

    alpha = torch.clip(alpha, min=clip_value, max=1.0)
    alpha_bar = torch.cumprod(alpha, axis=0)

    return alpha_bar
