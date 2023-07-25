import unittest

import torch

from diffusion_hopping.model.egnn.equivariant_gcl import EquivariantGCL


class EquivariantGCLTestEquivariance(unittest.TestCase):
    def test_equivariance(self):
        torch.manual_seed(0)
        hidden_features = 3
        edge_features = 4
        use_tanh = True
        gcl = EquivariantGCL(hidden_features, edge_features, use_tanh=use_tanh)
        x = torch.randn(10, hidden_features)
        pos = torch.randn(10, 3)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
        )
        mask = torch.tensor([True] * 5 + [False] * 5)
        edge_attr = torch.randn(10, edge_features)

        R, _ = torch.linalg.qr(torch.randn(3, 3))

        pos1 = gcl(x, pos, edge_index, mask, edge_attr)
        pos2 = gcl(x, pos @ R, edge_index, mask, edge_attr)
        self.assertTrue(torch.allclose(pos1 @ R, pos2))


if __name__ == "__main__":
    unittest.main()
