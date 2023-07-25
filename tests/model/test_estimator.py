import unittest

import torch
from torch_geometric.data import HeteroData

from diffusion_hopping.model.enum import Architecture
from diffusion_hopping.model.estimator import EstimatorModel


class TestEstimatorModel(unittest.TestCase):
    def test_estimator_embedding_initialization(self):
        model = EstimatorModel(
            ligand_features=3,
            protein_features=4,
            joint_features=5,
            architecture=Architecture.EGNN,
            hidden_features=8,
            edge_cutoff=1.0,
            egnn_velocity_parametrization=True,
            num_layers=6,
        )

        self.assertEqual(model.atom_encoder[0].in_features, 3)
        self.assertEqual(model.atom_encoder[0].out_features, 6)
        self.assertEqual(model.atom_encoder[2].out_features, 5)

        self.assertEqual(model.atom_decoder[0].in_features, 5)
        self.assertEqual(model.atom_decoder[0].out_features, 6)
        self.assertEqual(model.atom_decoder[2].out_features, 3)

        self.assertEqual(model.functional_group_encoder[0].in_features, 3)
        self.assertEqual(model.functional_group_encoder[0].out_features, 6)
        self.assertEqual(model.functional_group_encoder[2].out_features, 5)

        self.assertEqual(model.residue_encoder[0].in_features, 4)
        self.assertEqual(model.residue_encoder[0].out_features, 8)
        self.assertEqual(model.residue_encoder[2].out_features, 5)

    def test_estimator_forward_egnn(self):
        torch.manual_seed(0)
        model = EstimatorModel(
            ligand_features=3,
            protein_features=4,
            joint_features=5,
            architecture=Architecture.EGNN,
            hidden_features=8,
            edge_cutoff=1.0,
            egnn_velocity_parametrization=True,
            num_layers=6,
        )

        ligand_features = torch.randn(2, 3)
        ligand_pos = torch.randn(2, 3)
        protein_features = torch.randn(2, 4)
        protein_pos = torch.randn(2, 3)
        t = torch.tensor(0.2)
        batch = torch.tensor([0, 0])
        ligand_mask = torch.tensor([True, False])
        x_t = HeteroData(
            ligand={
                "x": ligand_features,
                "pos": ligand_pos,
                "batch": batch,
            },
            protein={
                "x": protein_features,
                "pos": protein_pos,
                "batch": batch,
            },
        )
        x_eps, pos_eps = model(x_t, t, ligand_mask)

        self.assertEqual(x_eps.shape, (1, 3))
        self.assertEqual(x_eps.shape, (1, 3))

    def test_estimator_forward_gvp(self):
        torch.manual_seed(0)
        model = EstimatorModel(
            ligand_features=3,
            protein_features=4,
            joint_features=5,
            architecture=Architecture.GVP,
            hidden_features=8,
            edge_cutoff=1.0,
            egnn_velocity_parametrization=True,
            num_layers=6,
        )

        ligand_features = torch.randn(2, 3)
        ligand_pos = torch.randn(2, 3)
        protein_features = torch.randn(2, 4)
        protein_pos = torch.randn(2, 3)
        t = torch.tensor([[0.2], [0.4]])
        batch = torch.tensor([0, 1])
        ligand_mask = torch.tensor([True, True])
        x_t = HeteroData(
            ligand={
                "x": ligand_features,
                "pos": ligand_pos,
                "batch": batch,
            },
            protein={
                "x": protein_features,
                "pos": protein_pos,
                "batch": batch,
            },
        )
        x_eps, pos_eps = model(x_t, t, ligand_mask)

        self.assertEqual(x_eps.shape, (2, 3))
        self.assertEqual(x_eps.shape, (2, 3))
