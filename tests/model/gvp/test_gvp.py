import unittest

import torch

from diffusion_hopping.model.gvp.gvp import GVP


class GVPTest(unittest.TestCase):
    def test_gvp_equivariance(self):
        torch.manual_seed(0)
        in_dims = (3, 4)
        out_dims = (5, 6)
        vector_gate = True
        gvp = GVP(in_dims, out_dims, vector_gate=vector_gate).double()
        s = torch.randn(10, 3, dtype=torch.float64)
        V = torch.randn(10, 4, 3, dtype=torch.float64)
        x1 = (s, V)
        R, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        x2 = (s, V @ R)
        s_dash1, V_dash1 = gvp(x1)
        s_dash2, V_dash2 = gvp(x2)

        self.assertEqual(s_dash1.shape, (10, 5))
        self.assertEqual(V_dash1.shape, (10, 6, 3))
        self.assertEqual(s_dash2.shape, (10, 5))
        self.assertEqual(V_dash2.shape, (10, 6, 3))
        self.assertTrue(torch.allclose(s_dash1, s_dash2))
        self.assertTrue(torch.allclose(V_dash1 @ R, V_dash2))

    def test_gvp_no_vector_gate_equivariance(self):
        torch.manual_seed(0)
        in_dims = (3, 4)
        out_dims = (5, 6)
        vector_gate = False
        gvp = GVP(in_dims, out_dims, vector_gate=vector_gate).double()
        s = torch.randn(10, 3, dtype=torch.float64)
        V = torch.randn(10, 4, 3, dtype=torch.float64)
        x1 = (s, V)
        R, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        x2 = (s, V @ R)
        s_dash1, V_dash1 = gvp(x1)
        s_dash2, V_dash2 = gvp(x2)

        self.assertEqual(s_dash1.shape, (10, 5))
        self.assertEqual(V_dash1.shape, (10, 6, 3))
        self.assertEqual(s_dash2.shape, (10, 5))
        self.assertEqual(V_dash2.shape, (10, 6, 3))
        self.assertTrue(torch.allclose(s_dash1, s_dash2))
        self.assertTrue(torch.allclose(V_dash1 @ R, V_dash2))


if __name__ == "__main__":
    unittest.main()
