import warnings

import torch
import torch_scatter
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data as DataBatch
from tqdm import tqdm

from diffusion_hopping.model import util as util
from diffusion_hopping.model.diffusion.schedules import PolynomialBetaSchedule
from diffusion_hopping.model.enum import Parametrization, SamplingMode


class DiffusionModel(nn.Module):
    def __init__(
        self,
        estimator: nn.Module,
        T: int = 200,
        parametrization=Parametrization.EPS,
        pos_norm=1.0,
        x_norm=1.0,
        x_bias=0.0,
        condition_on_fg=True,
        schedule=PolynomialBetaSchedule,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.parametrization = parametrization
        self.schedule = schedule(T)
        self.T = T

        self.pos_norm = pos_norm
        self.x_norm = x_norm
        self.x_bias = x_bias
        self.condition_on_fg = condition_on_fg

    def get_mask(self, x_0: DataBatch):
        if self.condition_on_fg:
            return x_0["ligand"].scaffold_mask
        else:
            return torch.ones_like(x_0["ligand"].scaffold_mask, dtype=torch.bool)

    def sample_times(self, batch_size, device):
        return torch.randint(0, self.T, (batch_size, 1), device=device)

    def q(self, x_0: DataBatch, t, mask=None, return_eps=False):
        device = x_0["ligand"].pos.device
        if mask is None:
            mask = torch.ones(x_0["ligand"].num_nodes, dtype=torch.bool, device=device)

        x_eps = torch.randn_like(x_0["ligand"].x[mask], device=device)
        # shape: (masked_nodes, num_features)
        pos_eps = util.centered_batch(
            torch.randn_like(x_0["ligand"].pos[mask], device=device),
            x_0["ligand"].batch[mask],
            dim_size=x_0.num_graphs,
        )  # shape: (masked_nodes, 3)
        if isinstance(t, torch.Tensor) and torch.numel(t) > 1:
            t = t[x_0["ligand"].batch[mask]]  # shape: (masked_nodes, 1)
        # otherwise, t is a scalar

        # shape: (masked_nodes, 1) afterwards or () if t is a scalar
        sqrt_alpha_bar = self.schedule.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alpha_bar[t]

        # noise x
        x_t = x_0.clone()
        # X ~ N[sqrt(alpha_bar)*x_0, 1 - alpha_bar]
        x_t["ligand"].x[mask] = (
            sqrt_alpha_bar * x_0["ligand"].x[mask] + sqrt_one_minus_alpha_bar * x_eps
        )
        x_t["ligand"].pos[mask] = (
            sqrt_alpha_bar * x_0["ligand"].pos[mask]
            + sqrt_one_minus_alpha_bar * pos_eps
        )
        if return_eps:
            return x_t, x_eps, pos_eps
        else:
            return x_t

    def centered_complex(self, x_t, mask=None):
        x_t = x_t.clone()
        pos = x_t["ligand"].pos
        batch = x_t["ligand"].batch
        if mask is not None:
            pos = pos[mask]
            batch = batch[mask]
        mean = torch_scatter.scatter_mean(
            pos,
            batch,
            dim=0,
            dim_size=x_t.num_graphs,
        )
        x_t["ligand"].pos -= mean[x_t["ligand"].batch]
        x_t["protein"].pos -= mean[x_t["protein"].batch]
        return x_t

    def uncentered_complex(self, x_t, mean):
        x_t = x_t.clone()
        x_t["ligand"].pos += mean[x_t["ligand"].batch]
        x_t["protein"].pos += mean[x_t["protein"].batch]
        return x_t

    def normalize(self, x_t):
        x_t = x_t.clone()
        x_t["ligand"].pos /= self.pos_norm
        x_t["protein"].pos /= self.pos_norm
        x_t["ligand"].x = x_t["ligand"].x.float() / self.x_norm + self.x_bias
        x_t["protein"].x = x_t["protein"].x.float() / self.x_norm + self.x_bias
        return x_t

    def denormalize(self, x_t):
        x_t = x_t.clone()
        x_t["ligand"].pos *= self.pos_norm
        x_t["protein"].pos *= self.pos_norm
        x_t["ligand"].x = (x_t["ligand"].x - self.x_bias) * self.x_norm
        x_t["protein"].x = (x_t["protein"].x - self.x_bias) * self.x_norm
        return x_t

    def estimate_mu(self, x_t, t, mask, x_eps, pos_eps):
        t = t[x_t["ligand"].batch[mask]]
        mu_x_t = self.schedule.sqrt_recip_alpha[t] * (
            x_t["ligand"].x[mask]
            - self.schedule.beta[t] / self.schedule.sqrt_one_minus_alpha_bar[t] * x_eps
        )
        mu_pos_t = self.schedule.sqrt_recip_alpha[t] * (
            x_t["ligand"].pos[mask]
            - self.schedule.beta[t]
            / self.schedule.sqrt_one_minus_alpha_bar[t]
            * pos_eps
        )
        return mu_x_t, mu_pos_t

    def forward(
        self,
        x_0,
    ):

        mask = self.get_mask(x_0)
        x_0 = self.centered_complex(x_0, mask)
        x_0 = self.normalize(x_0)
        device = x_0["ligand"].pos.device

        # t ~ UniformInteger[0, T]
        t = self.sample_times(x_0.num_graphs, device)
        x_t, x_eps, pos_eps = self.q(x_0, t, mask, return_eps=True)

        if self.parametrization == Parametrization.EPS:
            x_eps_pred, pos_eps_pred = self.estimator(x_t, t / self.T, mask)
            L_pos = F.mse_loss(pos_eps, pos_eps_pred)
            L_x = F.mse_loss(x_eps, x_eps_pred)
            L_simple = 0.25 * (L_pos + L_x)
            eps = torch.cat((x_eps.flatten(), pos_eps.flatten()))
            eps_pred = torch.cat((x_eps_pred.flatten(), pos_eps_pred.flatten()))
            L_unweighted = 0.5 * F.mse_loss(eps_pred, eps)
            return (
                L_simple,
                L_unweighted,
                L_pos,
                L_x,
            )
        elif self.parametrization == Parametrization.MEAN:
            mu_x_t, mu_pos_t = self.estimate_mu(x_t, t, mask, x_eps, pos_eps)
            mu_x_pred, mu_pos_pred = self.estimator(x_t, t / self.T, mask)
            L_pos = F.mse_loss(mu_pos_t, mu_pos_pred)
            L_x = F.mse_loss(mu_x_t, mu_x_pred)
            L_simple = 0.25 * (L_pos + L_x)
            mu = torch.cat((mu_x_t.flatten(), mu_pos_t.flatten()))
            mu_pred = torch.cat((mu_x_pred.flatten(), mu_pos_pred.flatten()))
            L_unweighted = 0.5 * F.mse_loss(mu_pred, mu)
            return (
                L_simple,
                L_unweighted,
                L_pos,
                L_x,
            )
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def p(self, x_t, t, mask=None, add_noise=True):
        if mask is None:
            mask = torch.ones(
                x_t["ligand"].num_nodes,
                dtype=torch.bool,
                device=x_t["ligand"].pos.device,
            )

        x_pred, pos_pred = self.estimator(x_t, t / self.T, mask)
        x_s = x_t.clone()

        if self.parametrization == Parametrization.EPS:

            beta_t = self.schedule.beta[t[mask]]
            sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alpha_bar[t[mask]]
            sqrt_recip_alpha_t = self.schedule.sqrt_recip_alpha[t[mask]]

            x_s["ligand"].x[mask] = sqrt_recip_alpha_t * (
                x_t["ligand"].x[mask] - beta_t * x_pred / sqrt_one_minus_alpha_bar_t
            )
            x_s["ligand"].pos[mask] = sqrt_recip_alpha_t * (
                x_t["ligand"].pos[mask] - beta_t * pos_pred / sqrt_one_minus_alpha_bar_t
            )
        elif self.parametrization == Parametrization.MEAN:

            x_s["ligand"].x[mask] = x_pred
            x_s["ligand"].pos[mask] = pos_pred

        else:
            raise NotImplementedError()

        x_s["ligand"].pos[mask] = util.centered_batch(
            x_s["ligand"].pos[mask],
            x_s["ligand"].batch[mask],
            dim_size=x_s.num_graphs,
        )
        # or:
        # x_s = self._center_protein_ligand_complex(x_s, mask)

        if add_noise:
            posterior_variance_t = self.schedule.posterior_variance[t[mask]]
            x_noise = torch.randn_like(x_s["ligand"].x[mask])
            x_s["ligand"].x[mask] += torch.sqrt(posterior_variance_t) * x_noise

            pos_noise = util.centered_batch(
                torch.randn_like(
                    x_s["ligand"].pos[mask],
                    device=x_s["ligand"].pos.device,
                ),
                x_s["ligand"].batch[mask],
                dim_size=x_s.num_graphs,
            )
            x_s["ligand"].pos[mask] += torch.sqrt(posterior_variance_t) * pos_noise

        return x_s

    def x_T_from_x_0(self, x_0, mask=None):
        device = x_0["ligand"].x.device
        x_T = x_0.clone()
        if mask is None:
            mask = torch.ones(x_0["ligand"].num_nodes, dtype=torch.bool, device=device)

        x_T["ligand"].x[mask] = torch.randn_like(x_T["ligand"].x[mask], device=device)
        x_T["ligand"].pos[mask] = util.centered_batch(
            torch.randn_like(x_T["ligand"].pos[mask], device=device),
            x_T["ligand"].batch[mask],
            dim_size=x_T.num_graphs,
        )

        return x_T

    @torch.no_grad()
    def sample(self, x_0, mode: SamplingMode = SamplingMode.DDPM):
        device = x_0["ligand"].x.device

        mask = self.get_mask(x_0)
        mean = torch_scatter.scatter_mean(
            x_0["ligand"].pos[mask],
            x_0["ligand"].batch[mask],
            dim=0,
            dim_size=x_0.num_graphs,
        )

        x_0 = self.centered_complex(x_0, mask)
        x_0 = self.normalize(x_0)
        x_t = self.x_T_from_x_0(x_0, mask)
        x = [self.uncentered_complex(self.denormalize(x_t.detach()), mean=mean).cpu()]
        if mode == SamplingMode.DDPM:
            noise_lambda = lambda t: t > 0
        elif mode == SamplingMode.DDIM:
            noise_lambda = lambda t: False
        else:
            raise NotImplementedError("mode must be either DDPM or DDIM.")

        for i in tqdm(reversed(range(0, self.T)), total=self.T):
            t = torch.full(
                (x_t["ligand"].num_nodes, 1), i, device=device, dtype=torch.long
            )
            x_t = self.p(x_t, t, mask, add_noise=noise_lambda(i))
            if x_t["ligand"].x[mask].isnan().any():
                print(f"NaNs in x_t after step {i}")
            x.append(
                self.uncentered_complex(self.denormalize(x_t.detach()), mean=mean).cpu()
            )

        return x

    def sample_x_t_plus_one(self, x_t, t, mask=None):
        if t.max() >= self.T or t.min() < 0:
            raise ValueError(f"t must be in range [0, {self.T})")
        if mask is None:
            mask = torch.ones(
                x_t["ligand"].num_nodes,
                dtype=torch.bool,
                device=x_t["ligand"].pos.device,
            )

        x_t_plus_one = x_t.clone()
        beta = self.schedule.beta[t]
        sqrt_alpha = self.schedule.sqrt_alpha[t]

        eps_x = torch.randn_like(x_t["ligand"].x[mask])
        x_t_plus_one["ligand"].x[mask] = (
            sqrt_alpha * x_t["ligand"].x[mask] + torch.sqrt(beta) * eps_x
        )

        eps_pos = util.centered_batch(
            torch.randn_like(x_t["ligand"].pos[mask]),
            x_t["ligand"].batch,
            dim_size=x_t.num_graphs,
        )
        x_t_plus_one["ligand"].pos[mask] = (
            sqrt_alpha * x_t["ligand"].pos[mask] + torch.sqrt(beta) * eps_pos
        )

        return x_t_plus_one

    def _repaint_schedule(
        self,
        j=10,
        r=10,
    ):

        jump_len = j
        jump_n_sample = r
        jumps = {j: jump_n_sample - 1 for j in range(0, self.T - jump_len, jump_len)}

        ts = [self.T - 1]

        while ts[-1] >= 0:
            if jumps.get(ts[-1], 0) > 0:
                jumps[ts[-1]] -= 1
                ts.extend(range(ts[-1] + 1, ts[-1] + jump_len + 1))
            ts.append(ts[-1] - 1)
        return ts

    def _merge_inpainting_while_keeping_centering(
        self, x_t_known, x_t_unknown, inpaint_mask
    ):
        x_t = x_t_known.clone()
        x_t["ligand"].x[inpaint_mask] = x_t_unknown["ligand"].x[inpaint_mask]
        x_t["ligand"].pos[inpaint_mask] = x_t_unknown["ligand"].pos[inpaint_mask]

        num_inpainted = torch.clip(
            torch_scatter.scatter_add(
                inpaint_mask.long(),
                x_t["ligand"].batch,
                dim=0,
                dim_size=x_t.num_graphs,
            ),
            min=1,
        )

        # Moving the inpainted nodes to ensure a centre of mass free system without moving the nodes given.
        offset = (
            torch_scatter.scatter_add(
                x_t["ligand"].pos,
                x_t["ligand"].batch,
                dim=0,
                dim_size=x_t.num_graphs,
            )
            / num_inpainted[:, None]
        )

        x_t["ligand"].pos[inpaint_mask] -= offset[x_t["ligand"].batch[inpaint_mask]]

        mean_post_adjustment = torch_scatter.scatter_mean(
            x_t["ligand"].pos,
            x_t["ligand"].batch,
            dim=0,
            dim_size=x_t.num_graphs,
        )
        if mean_post_adjustment.abs().max() > 1e-3:

            warnings.warn(
                f"Mean of inpainted system is not zero, it is {mean_post_adjustment}. "
                f"This is likely due to numerical errors."
            )

        return x_t

    def inpaint(self, x_0, inpaint_mask, j=10, r=10, center_input=True):
        device = x_0["ligand"].x.device
        if not self.get_mask(x_0).all():
            raise ValueError("To use inpaint, model has to be trained without masking")

        mean = torch_scatter.scatter_mean(
            x_0["ligand"].pos,
            x_0["ligand"].batch,
            dim=0,
            dim_size=x_0.num_graphs,
        )

        if center_input:
            x_0 = self.centered_complex(x_0)

        x_0 = self.normalize(x_0)
        x_t = self.x_T_from_x_0(x_0)
        x = [self.uncentered_complex(self.denormalize(x_t.detach()), mean=mean).cpu()]

        schedule = self._repaint_schedule(j=j, r=r)

        for i_last, i_cur in tqdm(list(zip(schedule[:-1], schedule[1:]))):
            t_last = torch.full(
                (x_t["ligand"].num_nodes, 1), i_last, device=device, dtype=torch.long
            )
            t_cur = torch.full(
                (x_t["ligand"].num_nodes, 1), i_cur, device=device, dtype=torch.long
            )
            if i_cur < i_last:
                # x_t_unknown = p(x_{t+1}, t + 1)
                x_t_unknown = self.p(x_t, t_last, add_noise=i_last > 0)
                # x_t_known = q(x_0, t)
                x_t_known = (
                    self.q(x_0, t_cur) if i_cur >= 0 else x_0
                )  # x_0 if i_cur == -1

                x_t = self._merge_inpainting_while_keeping_centering(
                    x_t_known, x_t_unknown, inpaint_mask
                )
            else:
                # x_t ~ N(sqrt(1 - beta_{t}) * x_{t-1}, beta_{t})
                x_t = self.sample_x_t_plus_one(x_t, t_cur)
            x.append(
                self.uncentered_complex(self.denormalize(x_t.detach()), mean=mean).cpu()
            )
        return x
