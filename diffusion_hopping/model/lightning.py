from collections import deque
from typing import Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from rdkit.Chem import Draw
from torch_geometric.data import HeteroData
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from diffusion_hopping.analysis.build import MoleculeBuilder
from diffusion_hopping.analysis.metrics import (
    MolecularConnectivity,
    MolecularLipinski,
    MolecularLogP,
    MolecularNovelty,
    MolecularQEDValue,
    MolecularSAScore,
    MolecularValidity,
)
from diffusion_hopping.model.diffusion.model import DiffusionModel
from diffusion_hopping.model.enum import Architecture, Parametrization
from diffusion_hopping.model.estimator import EstimatorModel
from diffusion_hopping.model.util import skip_computation_on_oom

image_to_tensor = ToTensor()


class DiffusionHoppingModel(pl.LightningModule):
    def __init__(
        self,
        # Diffusion parameters
        T=500,
        parametrization=Parametrization.EPS,
        # Training parameters
        lr=1e-4,
        clip_grad=False,
        condition_on_fg=False,
        # Normalization parameters
        pos_norm=1.0,
        x_norm=1.0,
        x_bias=0.0,
        # Estimator parameters
        architecture: Architecture = Architecture.EGNN,
        edge_cutoff=None,
        hidden_features=256,
        joint_features=32,
        num_layers=6,
        attention=True,
        # Dataset parameters
        ligand_features=10,
        protein_features=20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.atom_features = 10
        self.c_alpha_features = 20

        model_params = dict(
            hidden_features=hidden_features,
            num_layers=num_layers,
            attention=attention,
        )

        estimator = EstimatorModel(
            ligand_features=ligand_features,
            protein_features=protein_features,
            joint_features=joint_features,
            edge_cutoff=edge_cutoff,
            architecture=architecture,
            egnn_velocity_parametrization=(parametrization == Parametrization.EPS),
            **model_params,
        )

        self.model = DiffusionModel(
            estimator,
            T=T,
            parametrization=parametrization,
            pos_norm=pos_norm,
            x_norm=x_norm,
            x_bias=x_bias,
            condition_on_fg=condition_on_fg,
        )

        self.lr = lr
        self.clip_grad = clip_grad
        if self.clip_grad:
            self.gradient_norm_queue = deque([3000.0], maxlen=50)
        self.validation_metrics = None
        self.molecule_builder = MoleculeBuilder(include_invalid=True)

        self.analyse_samples_every_n_steps = 25000
        self.next_analyse_samples = self.analyse_samples_every_n_steps
        self._run_validation = False

    def setup_metrics(self, train_smiles):
        self.validation_metrics = torch.nn.ModuleDict(
            {
                "Novelty": MolecularNovelty(train_smiles),
                "Validity": MolecularValidity(),
                "Connectivity": MolecularConnectivity(),
                "Lipinski": MolecularLipinski(),
                "LogP": MolecularLogP(),
                "QED": MolecularQEDValue(),
                "SAScore": MolecularSAScore(),
            }
        )

    @skip_computation_on_oom(
        return_value=None, error_message="Skipping batch due to OOM"
    )
    def training_step(self, batch, batch_idx):
        (
            loss,
            loss_unweighted,
            pos_mse,
            x_mse,
        ) = self.model(batch)
        self.log("loss/train", loss, batch_size=batch.num_graphs)
        self.log("pos_mse/train", pos_mse, batch_size=batch.num_graphs)
        self.log("x_mse/train", x_mse, batch_size=batch.num_graphs)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._run_validation = self.global_step > self.next_analyse_samples

    def validation_step(self, batch, batch_idx):
        (
            loss,
            loss_unweighted,
            pos_mse,
            x_mse,
        ) = self.model(batch)
        self.log("loss/val", loss, batch_size=batch.num_graphs, sync_dist=True)
        self.log("pos_mse/val", pos_mse, batch_size=batch.num_graphs, sync_dist=True)
        self.log("x_mse/val", x_mse, batch_size=batch.num_graphs, sync_dist=True)
        if self._run_validation:
            self.analyse_samples(batch, batch_idx)

        return loss

    def analyse_samples(self, batch, batch_idx):

        samples = self.model.sample(batch)[-1]

        molecules = self.molecule_builder(samples)

        for k, metric in self.validation_metrics.items():
            metric(molecules)
            self.log(
                f"{k}/val",
                metric,
                batch_size=batch.num_graphs,
                sync_dist=True,
            )
        self.log_molecule_visualizations(molecules, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if self._run_validation:
            self.next_analyse_samples = (
                self.global_step
                - (self.global_step % self.analyse_samples_every_n_steps)
                + self.analyse_samples_every_n_steps
            )

    def log_molecule_visualizations(self, molecules, batch_idx):
        images = []
        captions = []
        for i, mol in enumerate(molecules):
            if mol is None:
                continue
            img = image_to_tensor(Draw.MolToImage(mol, size=(500, 500)))
            images.append(img)
            captions.append(f"{self.current_epoch}_{batch_idx}_{i}")

        grid_image = make_grid(images)
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_image(
                    f"log_image_{batch_idx}",
                    grid_image,
                    self.current_epoch,
                )
            if isinstance(logger, pl.loggers.WandbLogger):
                logger.log_image(key="test_set_images", images=images, caption=captions)
        # log_dir = Path(self.logger.log_dir) / "samples"
        # log_dir.mkdir(exist_ok=True, parents=True)
        # Draw.MolToFile(mol, f"{log_dir}/{self.current_epoch}_{batch_idx}_{i}.png")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12
        )
        return optimizer

    def configure_gradient_clipping(
        self,
        optimizer: optim.Optimizer,
        optimizer_idx: int,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * (standard deviation of the recent history).
        max_grad_norm: float = 1.5 * np.mean(self.gradient_norm_queue) + 2 * np.std(
            self.gradient_norm_queue
        )

        # Get current grad_norm
        grad_norm = float(get_grad_norm(optimizer))

        self.gradient_norm_queue.append(min(grad_norm, max_grad_norm))

        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        """Override this method to change the default behaviour of ``log_grad_norm``.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        Example::

            # DEFAULT
            def log_grad_norm(self, grad_norm_dict):
                self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        """
        results = self.trainer._results
        if isinstance(results.batch, HeteroData):
            results.batch_size = results.batch.num_graphs
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )


def get_grad_norm(
    optimizer: torch.optim.Optimizer, norm_type: float = 2.0
) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )

    return total_norm
