import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from diffusion_hopping.model.diffusion.model import DiffusionModel


class TestDiffusionModel(unittest.TestCase):
    def test_get_mask1(self):
        estimator = MagicMock()
        diffusion_model = DiffusionModel(estimator, condition_on_fg=True)
        mask_pre = torch.tensor([0, 0, 0, 1], dtype=torch.bool)
        x_0 = MagicMock(
            __getitem__=MagicMock(return_value=MagicMock(scaffold_mask=mask_pre))
        )
        mask = diffusion_model.get_mask(x_0)
        self.assertTrue((mask == mask_pre).all())

    def test_get_mask2(self):
        estimator = MagicMock()
        diffusion_model = DiffusionModel(estimator, condition_on_fg=False)
        mask_pre = torch.tensor([0, 0, 0, 1], dtype=torch.bool)
        mask_post = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
        x_0 = MagicMock(
            __getitem__=MagicMock(return_value=MagicMock(scaffold_mask=mask_pre))
        )
        mask = diffusion_model.get_mask(x_0)
        self.assertTrue((mask == mask_post).all())


if __name__ == "__main__":
    unittest.main()
