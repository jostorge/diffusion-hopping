import unittest

import torch

from diffusion_hopping.model.egnn import EGNN


class EGNNTestEquivariance(unittest.TestCase):
    def test_equivariance1(self):
        torch.manual_seed(0)

        in_features = 3
        out_features = 4
        edge_features = 5
        hidden_features = 6
        num_layers = 7
        attention = False
        use_tanh = True
        egnn = EGNN(
            in_features,
            out_features,
            edge_features,
            hidden_features,
            num_layers,
            attention,
            use_tanh,
        )
        x = torch.randn(10, in_features)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
        )
        mask = torch.tensor([True] * 10)
        edge_attr = torch.randn(10, edge_features)

        R, _ = torch.linalg.qr(torch.randn(3, 3))

        x1, pos1 = egnn(x, pos, edge_index, mask, edge_attr)
        x2, pos2 = egnn(x, pos @ R, edge_index, mask, edge_attr)
        self.assertTrue(torch.allclose(x1, x2))
        self.assertTrue(torch.allclose(pos1 @ R, pos2))

    def test_equivariance2(self):
        torch.manual_seed(1)

        in_features = 7
        out_features = 6
        edge_features = 5
        hidden_features = 4
        num_layers = 3
        attention = True
        use_tanh = False
        egnn = EGNN(
            in_features,
            out_features,
            edge_features,
            hidden_features,
            num_layers,
            attention,
            use_tanh,
        )
        x = torch.randn(10, in_features)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
        )
        mask = torch.tensor([True] * 5 + [False] * 5)
        edge_attr = torch.randn(10, edge_features)

        R, _ = torch.linalg.qr(torch.randn(3, 3))

        x1, pos1 = egnn(x, pos, edge_index, mask, edge_attr)
        x2, pos2 = egnn(x, pos @ R, edge_index, mask, edge_attr)
        self.assertTrue(torch.allclose(x1, x2))
        self.assertTrue(torch.allclose(pos1 @ R, pos2))


if __name__ == "__main__":
    unittest.main()
