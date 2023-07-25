import numpy as np
import torch
import torch.nn as nn

from diffusion_hopping.model.egnn import EGNN
from diffusion_hopping.model.enum import Architecture
from diffusion_hopping.model.gvp import GVPNetwork


class EstimatorModel(nn.Module):
    def __init__(
        self,
        ligand_features,
        protein_features,
        architecture: Architecture = Architecture.GVP,
        joint_features=16,
        edge_cutoff=None,
        egnn_velocity_parametrization=True,
        **kwargs,
    ):
        super().__init__()
        if edge_cutoff is None or isinstance(edge_cutoff, (int, float)):
            edge_cutoff = (edge_cutoff, edge_cutoff, edge_cutoff)

        (
            self.edge_cutoff_ligand,
            self.edge_cutoff_protein,
            self.edge_cutoff_cross,
        ) = edge_cutoff

        self.egnn_velocity_parametrization = egnn_velocity_parametrization
        self.architecture = architecture

        self.atom_encoder = nn.Sequential(
            nn.Linear(ligand_features, 2 * ligand_features),
            nn.SiLU(),
            nn.Linear(2 * ligand_features, joint_features),
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_features, 2 * ligand_features),
            nn.SiLU(),
            nn.Linear(2 * ligand_features, ligand_features),
        )

        self.functional_group_encoder = nn.Sequential(
            nn.Linear(ligand_features, 2 * ligand_features),
            nn.SiLU(),
            nn.Linear(2 * ligand_features, joint_features),
        )
        self.residue_encoder = nn.Sequential(
            nn.Linear(protein_features, 2 * protein_features),
            nn.SiLU(),
            nn.Linear(2 * protein_features, joint_features),
        )

        dynamics_node_nf = joint_features + 1
        if architecture == Architecture.GVP:
            hidden_dims = (
                kwargs["hidden_features"] // 2,
                kwargs["hidden_features"] // 2,
            )
            del kwargs["hidden_features"]
            self.gvp = GVPNetwork(
                in_dims=(dynamics_node_nf, 0),
                out_dims=(joint_features, 1),
                hidden_dims=hidden_dims,
                vector_gate=True,
                **kwargs,
            )

        elif architecture == Architecture.EGNN:
            self.egnn = EGNN(
                in_features=dynamics_node_nf,
                out_features=joint_features,
                edge_features=0,
                **kwargs,
            )
        else:
            raise ValueError("Unknown mode")

    def forward(self, x_t, t, ligand_mask):
        batch_ligand = x_t["ligand"].batch
        batch_protein = x_t["protein"].batch

        pos_ligand = x_t["ligand"].pos
        pos_protein = x_t["protein"].pos

        x_ligand = x_t["ligand"].x
        x_protein = x_t["protein"].x

        # embed atom features and residue features in a shared space
        x_ligand = torch.where(
            ligand_mask[:, None],
            self.atom_encoder(x_ligand),
            self.functional_group_encoder(x_ligand),
        )
        x_protein = self.residue_encoder(x_protein)

        # combine the two node types
        pos = torch.cat((pos_ligand, pos_protein), dim=0)
        x = torch.cat((x_ligand, x_protein), dim=0)
        is_protein = torch.cat(
            (
                torch.zeros_like(batch_ligand, dtype=torch.bool),
                torch.ones_like(batch_protein, dtype=torch.bool),
            ),
            dim=0,
        )
        batch = torch.cat([batch_ligand, batch_protein])

        if np.prod(t.size()) == 1:
            # t is the same for all elements in batch.
            x_time = torch.empty_like(x[:, 0:1]).fill_(t.item())
        else:
            # t is different over the batch dimension.
            x_time = t[batch]

        x = torch.cat([x, x_time], dim=1)

        edge_index = self.get_edges(batch, pos, is_protein)

        if self.architecture == Architecture.EGNN:
            # update_coords_mask is a long tensor, not a bool tensor
            # this is intended, because the EGNN expects a long tensor
            update_coords_mask = torch.cat(
                (ligand_mask, torch.zeros_like(batch_protein))
            )

            x_final, pos_out = self.egnn(x, pos, edge_index, update_coords_mask)

            if self.egnn_velocity_parametrization:
                pos_out = pos_out - pos  # pos_out is now the velocity

            if torch.any(torch.isnan(pos_out)):
                print("Warning: detected nan, resetting EGNN output to zero.")
                pos_out = torch.zeros_like(pos_out)

        elif self.architecture == Architecture.GVP:
            x_final, pos_out = self.gvp(
                x,
                pos,
                edge_index,
            )
            # pos_out has shape [N, 1, 3]
            pos_out = pos_out.reshape(-1, 3)
        else:
            raise ValueError("Unknown mode")

        # decode atom features
        x_final_ligand = self.atom_decoder(x_final[: len(batch_ligand)])
        pos_out_ligand = pos_out[: len(batch_ligand)]

        return x_final_ligand[ligand_mask], pos_out_ligand[ligand_mask]

    def get_edges(self, batch_mask, x, is_protein):
        is_ligand = ~is_protein

        adj_batch = batch_mask[:, None] == batch_mask[None, :]
        protein_adj = is_protein[:, None] & is_protein[None, :]
        ligand_adj = is_ligand[:, None] & is_ligand[None, :]
        cross_adj = (is_protein[:, None] & is_ligand[None, :]) | (
            is_ligand[:, None] & is_protein[None, :]
        )
        if self.edge_cutoff_ligand is not None:
            ligand_adj = ligand_adj & (torch.cdist(x, x) < self.edge_cutoff_ligand)
        if self.edge_cutoff_protein is not None:
            protein_adj = protein_adj & (torch.cdist(x, x) < self.edge_cutoff_protein)
        if self.edge_cutoff_cross is not None:
            cross_adj = cross_adj & (torch.cdist(x, x) < self.edge_cutoff_cross)
        adj = adj_batch & (protein_adj | ligand_adj | cross_adj)
        edges = torch.stack(torch.where(adj), dim=0)
        return edges
