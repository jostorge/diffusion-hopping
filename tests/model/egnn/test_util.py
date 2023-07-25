import unittest

import torch

from diffusion_hopping.model.egnn.util import get_squared_distance


class UtilTest(unittest.TestCase):
    def test_get_squared_distance(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        squared_distance = get_squared_distance(pos, edge_index)
        self.assertTrue(torch.allclose(squared_distance, torch.tensor([[3.0], [3.0]])))
