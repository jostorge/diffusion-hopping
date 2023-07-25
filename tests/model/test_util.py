import unittest

import torch

from diffusion_hopping.model.util import centered_batch, skip_computation_on_oom


class TestCenteredBatch(unittest.TestCase):
    def test_centered_batch(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        batch = torch.tensor([0, 0, 1, 1])

        desired_output = torch.tensor([[-1, -1], [1, 1], [-1, -1], [1, 1]])
        output = centered_batch(x, batch)

        self.assertTrue(torch.allclose(output, desired_output))

    def test_centered_batch_with_mask(self):
        x = torch.tensor([[1, 2], [3, 4], [5, 6], [8, 9]])
        batch = torch.tensor([0, 0, 1, 1])
        mask = torch.tensor([True, False, True, False])

        desired_output = torch.tensor([[0, 0], [2, 2], [0, 0], [3, 3]])
        output = centered_batch(x, batch, mask=mask)
        self.assertTrue(torch.allclose(output, desired_output))


class TestSkipComputationOnOOM(unittest.TestCase):
    def test_skip_computation_on_oom(self):
        @skip_computation_on_oom(return_value=1)
        def func():
            raise RuntimeError("out of memory")

        self.assertEqual(func(), 1)

    def test_skip_computation_on_oom_with_error_message(self):
        @skip_computation_on_oom(return_value=1, error_message="error")
        def func():
            raise RuntimeError("out of memory")

        self.assertEqual(func(), 1)

    def test_skip_computation_on_oom_with_other_error(self):
        @skip_computation_on_oom(return_value=1, error_message="error")
        def func():
            raise RuntimeError("other error")

        with self.assertRaises(RuntimeError):
            func()

    def test_skip_computation_on_oom_without_error_message(self):
        @skip_computation_on_oom(return_value=1)
        def func():
            raise RuntimeError("out of memory")

        self.assertEqual(func(), 1)

    def test_skip_computation_on_oom_passthrough(self):
        @skip_computation_on_oom()
        def func():
            return 4

        self.assertEqual(func(), 4)
