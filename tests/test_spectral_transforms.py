import math
import unittest

import torch

from smo.experimental.dct_pure import create_dct_matrix, dct_2d, idct_2d
from smo.experimental.walsh_pure import fwht_2d, ifwht_2d


def _hadamard_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    # Sylvester construction: H_2n = [[H_n, H_n], [H_n, -H_n]]
    h = torch.tensor([[1.0]], dtype=dtype)
    size = 1
    while size < n:
        top = torch.cat((h, h), dim=1)
        bottom = torch.cat((h, -h), dim=1)
        h = torch.cat((top, bottom), dim=0)
        size *= 2
    return h


class DCTTransformTests(unittest.TestCase):
    def test_dct_matrix_is_orthonormal(self):
        d = create_dct_matrix(8, torch.float32, "cpu")
        identity = torch.eye(8)
        self.assertTrue(torch.allclose(d.t() @ d, identity, atol=1e-6))
        self.assertTrue(torch.allclose(d @ d.t(), identity, atol=1e-6))

    def test_dct_round_trip_reconstructs_input(self):
        torch.manual_seed(1234)
        x = torch.randn(8, 8)
        d = create_dct_matrix(8, torch.float32, "cpu")
        reconstructed = idct_2d(dct_2d(x, d, d), d, d)
        self.assertTrue(torch.allclose(x, reconstructed, atol=1e-5))

    def test_dct_preserves_energy(self):
        torch.manual_seed(1234)
        x = torch.randn(8, 8)
        d = create_dct_matrix(8, torch.float32, "cpu")
        x_dct = dct_2d(x, d, d)
        self.assertAlmostEqual(x.pow(2).sum().item(), x_dct.pow(2).sum().item(), places=3)

    def test_truncated_variance_reconstruction_can_go_negative(self):
        """Motivates the clamp(min=0) in SMODCTPure.step."""
        g = torch.randn(8, 8) * 10.0
        g[2:6, 2:6] = 0.0
        d = create_dct_matrix(8, torch.float32, "cpu")
        g_sq_dct = dct_2d(g.square(), d, d)
        truncated = torch.zeros_like(g_sq_dct)
        truncated[:4, :4] = g_sq_dct[:4, :4]
        g_sq_rec = idct_2d(truncated, d, d)
        self.assertLess(g_sq_rec.min().item(), 0.0)


class WalshTransformTests(unittest.TestCase):
    def test_fwht_matches_dense_hadamard(self):
        torch.manual_seed(1234)
        x = torch.randn(8, 8)
        h8 = _hadamard_matrix(8)
        expected = h8 @ x @ h8.t()
        self.assertTrue(torch.allclose(fwht_2d(x), expected, atol=1e-4))

    def test_ifwht_inverts_fwht(self):
        torch.manual_seed(1234)
        x = torch.randn(16, 16)
        # FWHT applied twice scales by numel; ifwht divides by numel once,
        # so fwht -> ifwht is the identity.
        round_trip = ifwht_2d(fwht_2d(x))
        self.assertTrue(torch.allclose(round_trip, x, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
