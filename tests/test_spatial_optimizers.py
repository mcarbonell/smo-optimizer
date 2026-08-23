import unittest

import torch

from smo.optimizers.spatial import SMO
from smo.optimizers.spatial_8bit import SMO8bit


class SpatialOptimizerTests(unittest.TestCase):
    def test_spatial_second_moment_tracks_pooled_squared_gradient(self):
        param = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.float32))
        optimizer = SMO([param], lr=1e-3, betas=(0.9, 0.999), k_ratio=0.25)

        grad = torch.empty_like(param)
        grad[:, ::2] = 10.0
        grad[:, 1::2] = -10.0
        param.grad = grad

        optimizer.step()

        state = optimizer.state[param]
        expected = SMO._compress_2d(grad.square(), state["exp_avg_sq"].shape) * (1 - 0.999)
        self.assertTrue(torch.allclose(state["exp_avg_sq"], expected, atol=1e-8, rtol=1e-5))
        self.assertGreater(state["exp_avg_sq"].mean().item(), 0.0)

    def test_spatial_and_8bit_match_when_quantization_is_exact(self):
        seed = 1234
        torch.manual_seed(seed)
        param_spatial = torch.nn.Parameter(torch.randn(32, 32, dtype=torch.float32))
        param_8bit = torch.nn.Parameter(param_spatial.detach().clone())
        grad = torch.randn_like(param_spatial)

        opt_spatial = SMO([param_spatial], lr=1e-3, betas=(0.9, 0.999), k_ratio=0.25)
        opt_8bit = SMO8bit([param_8bit], lr=1e-3, betas=(0.9, 0.999), k_ratio=0.25, block_size=1)

        param_spatial.grad = grad.clone()
        param_8bit.grad = grad.clone()

        opt_spatial.step()
        opt_8bit.step()

        state_spatial = opt_spatial.state[param_spatial]
        state_8bit = opt_8bit.state[param_8bit]
        dequant_v = opt_8bit._dequantize_blockwise(state_8bit["v_q"], state_8bit["v_s"], state_8bit["comp_shape"])

        self.assertTrue(torch.allclose(param_spatial, param_8bit, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(state_spatial["exp_avg_sq"], dequant_v, atol=1e-6, rtol=1e-5))

    def test_8bit_rejects_sparse_gradients(self):
        dense = torch.nn.Parameter(torch.zeros(8, 8, dtype=torch.float32))
        optimizer = SMO8bit([dense], lr=1e-3)
        indices = torch.tensor([[0, 1], [2, 3]])
        values = torch.tensor([1.0, -1.0])
        dense.grad = torch.sparse_coo_tensor(indices, values, dense.shape)

        with self.assertRaises(RuntimeError):
            optimizer.step()

    def test_compress_false_group_matches_dense_adam(self):
        torch.manual_seed(7)
        param_smo = torch.nn.Parameter(torch.randn(64, 64))
        param_adam = torch.nn.Parameter(param_smo.detach().clone())

        opt_smo = SMO([{"params": [param_smo], "compress": False}], lr=1e-3)
        opt_adam = torch.optim.Adam([param_adam], lr=1e-3)

        for _ in range(5):
            grad = torch.randn(64, 64)
            param_smo.grad = grad.clone()
            param_adam.grad = grad.clone()
            opt_smo.step()
            opt_adam.step()

        self.assertTrue(torch.allclose(param_smo, param_adam, atol=1e-6))
        self.assertFalse(opt_smo.state[param_smo]["is_compressed"])


class LowPeakBandedUpdateTests(unittest.TestCase):
    def _train_pair(self, shape, k_ratio=0.25, steps=3, weight_decay=0.0):
        torch.manual_seed(11)
        init = torch.randn(*shape)

        param_mono = torch.nn.Parameter(init.clone())
        param_band = torch.nn.Parameter(init.clone())
        opt_mono = SMO8bit([param_mono], lr=1e-3, k_ratio=k_ratio, block_size=64,
                           weight_decay=weight_decay)
        opt_band = SMO8bit([param_band], lr=1e-3, k_ratio=k_ratio, block_size=64,
                           weight_decay=weight_decay, low_peak=True, band_mb=0.05)

        torch.manual_seed(23)
        for _ in range(steps):
            grad = torch.randn_like(init)
            param_mono.grad = grad.clone()
            param_band.grad = grad.clone()
            opt_mono.step()
            opt_band.step()

        state_mono = opt_mono.state[param_mono]
        state_band = opt_band.state[param_band]
        return param_mono, param_band, state_mono, state_band

    def test_banded_matches_monolithic_on_exact_pool_shape(self):
        param_mono, param_band, state_mono, state_band = self._train_pair((128, 256))

        self.assertTrue(torch.allclose(param_mono, param_band, atol=1e-5))
        for key in ("m_q", "v_q"):
            self.assertTrue(torch.equal(state_mono[key], state_band[key]))
        for key in ("m_s", "v_s"):
            self.assertTrue(torch.allclose(state_mono[key], state_band[key]))

    def test_banded_falls_back_on_non_divisible_shape(self):
        # 50x30 with k=0.25 -> comp 12x7; 50 % 12 != 0 -> monolithic fallback
        param_mono, param_band, _, _ = self._train_pair((50, 30))
        self.assertTrue(torch.allclose(param_mono, param_band, atol=1e-6))


class PermuteBasisTests(unittest.TestCase):
    def test_permute_basis_is_identity_at_k1(self):
        # k=1 makes pooling/interpolation the identity and block_size=1 makes
        # quantization exact, so the basis permutation must cancel out.
        torch.manual_seed(5)
        init = torch.randn(64, 64)
        p_plain = torch.nn.Parameter(init.clone())
        p_perm = torch.nn.Parameter(init.clone())
        o_plain = SMO8bit([p_plain], lr=1e-3, k_ratio=1.0, block_size=1)
        o_perm = SMO8bit([p_perm], lr=1e-3, k_ratio=1.0, block_size=1, permute_basis=True)

        torch.manual_seed(9)
        for _ in range(3):
            grad = torch.randn(64, 64)
            p_plain.grad = grad.clone()
            p_perm.grad = grad.clone()
            o_plain.step()
            o_perm.step()

        self.assertTrue(torch.allclose(p_plain, p_perm, atol=1e-6))

    def test_permute_basis_changes_update_at_k025(self):
        torch.manual_seed(5)
        init = torch.randn(64, 64)
        p_plain = torch.nn.Parameter(init.clone())
        p_perm = torch.nn.Parameter(init.clone())
        o_plain = SMO8bit([p_plain], lr=1e-3, k_ratio=0.25)
        o_perm = SMO8bit([p_perm], lr=1e-3, k_ratio=0.25, permute_basis=True)

        torch.manual_seed(9)
        max_diff = 0.0
        for _ in range(3):
            grad = torch.randn(64, 64)
            p_plain.grad = grad.clone()
            p_perm.grad = grad.clone()
            o_plain.step()
            o_perm.step()
            max_diff = max(max_diff, (p_plain - p_perm).abs().max().item())

        self.assertGreater(max_diff, 1e-4)

    def test_permute_basis_rejects_low_peak(self):
        with self.assertRaises(ValueError):
            SMO8bit([torch.nn.Parameter(torch.zeros(32, 32))], low_peak=True, permute_basis=True)


if __name__ == "__main__":
    unittest.main()
