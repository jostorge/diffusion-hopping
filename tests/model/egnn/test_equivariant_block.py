import unittest

import torch

from diffusion_hopping.model.egnn.equivariant_block import EquivariantBlock


class EquivariantBlockTestEquivariance(unittest.TestCase):
    def test_equivariance(self):
        hidden_features = 3
        edge_features = 4
        num_layers = 4
        attention = False
        use_tanh = True
        eb = EquivariantBlock(
            hidden_features, edge_features, num_layers, attention, use_tanh
        )
        x = torch.randn(10, hidden_features)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
        )
        mask = torch.tensor([True] * 5 + [False] * 5)
        edge_attr = torch.randn(10, edge_features)

        R, _ = torch.linalg.qr(torch.randn(3, 3))

        x1, pos1 = eb(x, pos, edge_index, mask, edge_attr)
        x2, pos2 = eb(x, pos @ R, edge_index, mask, edge_attr)
        self.assertTrue(torch.allclose(x1, x2))
        self.assertTrue(torch.allclose(pos1 @ R, pos2))


if __name__ == "__main__":
    unittest.main()
