import unittest

import torch

from smo.optimizers._spatial_utils import compression_view
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


class ConvPoolingTests(unittest.TestCase):
    def test_compression_view_eligibility(self):
        # 2D matrices: unchanged rule
        self.assertEqual(compression_view((64, 64)), ((64, 64), (64, 64)))
        self.assertIsNone(compression_view((16, 16)))
        # Conv weights are opt-in: dense by default, row view with the flag
        self.assertIsNone(compression_view((256, 128, 3, 3)))
        self.assertEqual(compression_view((256, 128, 3, 3), include_conv=True),
                         ((256, 1152), (256, 128, 3, 3)))
        # Small conv heads fall back even when opted in
        self.assertIsNone(compression_view((16, 3, 3, 3), include_conv=True))
        # Other ranks are untouched
        self.assertIsNone(compression_view((8, 16, 32), include_conv=True))

    def _train_conv_pair(self, cls, shape, **kwargs):
        torch.manual_seed(3)
        p_conv = torch.nn.Parameter(torch.randn(*shape))
        p_flat = torch.nn.Parameter(p_conv.detach().clone().view(-1, int(torch.tensor(shape[1:]).prod())))
        opt_conv = cls([p_conv], lr=1e-3, k_ratio=0.25, compress_conv=True, **kwargs)
        opt_flat = cls([p_flat], lr=1e-3, k_ratio=0.25, **kwargs)

        torch.manual_seed(13)
        for _ in range(4):
            g = torch.randn_like(p_conv)
            p_conv.grad = g.clone()
            p_flat.grad = g.view(p_flat.shape).clone()
            opt_conv.step()
            opt_flat.step()
        return p_conv, p_flat, opt_conv

    def test_smo_conv_matches_equivalent_2d_parameter(self):
        p_conv, p_flat, opt_conv = self._train_conv_pair(SMO, (64, 8, 3, 3))

        state = opt_conv.state[p_conv]
        self.assertTrue(state["is_compressed"])
        self.assertEqual(tuple(state["exp_avg"].shape), (16, 18))
        self.assertTrue(torch.allclose(p_conv.view(64, 72), p_flat, atol=1e-6))
        self.assertTrue(torch.isfinite(p_conv).all())

    def test_8bit_conv_matches_equivalent_2d_parameter(self):
        # block_size == comp width so the flattened twin uses the SAME
        # row-aligned quantization blocks as the conv-view parameter
        p_conv, p_flat, opt_conv = self._train_conv_pair(SMO8bit, (64, 8, 3, 3), block_size=18)

        state = opt_conv.state[p_conv]
        self.assertTrue(state["is_compressed"])
        self.assertEqual(tuple(state["comp_shape"]), (16, 18))
        self.assertTrue(torch.allclose(p_conv.view(64, 72), p_flat, atol=1e-6))

    def test_small_conv_falls_back_to_dense(self):
        param = torch.nn.Parameter(torch.randn(16, 3, 3, 3))
        optimizer = SMO([param], lr=1e-3, k_ratio=0.25)
        param.grad = torch.randn_like(param)
        optimizer.step()

        state = optimizer.state[param]
        self.assertFalse(state["is_compressed"])
        self.assertEqual(tuple(state["exp_avg"].shape), tuple(param.shape))

    def test_conv_pooling_is_opt_in(self):
        # Eligible conv weights stay DENSE unless compress_conv=True: pooling
        # across flattened input channels measured negative on CNNs.
        torch.manual_seed(5)
        param = torch.nn.Parameter(torch.randn(64, 8, 3, 3))

        default_opt = SMO8bit([param], lr=1e-3, k_ratio=0.25)
        param.grad = torch.randn_like(param)
        default_opt.step()
        self.assertFalse(default_opt.state[param]["is_compressed"])

        opt_in = SMO([param], lr=1e-3, k_ratio=0.25, compress_conv=True)
        param.grad = torch.randn_like(param)
        opt_in.step()
        self.assertTrue(opt_in.state[param]["is_compressed"])

    def test_conv_second_moment_tracks_pooled_squared_gradient(self):
        param = torch.nn.Parameter(torch.zeros(64, 8, 3, 3))
        optimizer = SMO([param], lr=1e-3, betas=(0.9, 0.999), k_ratio=0.25,
                        compress_conv=True)

        grad = torch.empty_like(param)
        grad[:, :, :, ::2] = 10.0
        grad[:, :, :, 1::2] = -10.0
        param.grad = grad
        optimizer.step()

        state = optimizer.state[param]
        flat_sq = grad.view(64, 72).square()
        expected = SMO._compress_2d(flat_sq, state["exp_avg_sq"].shape) * (1 - 0.999)
        self.assertTrue(torch.allclose(state["exp_avg_sq"], expected, atol=1e-8, rtol=1e-5))


class ConvQuantizationPolicyTests(unittest.TestCase):
    def test_no_nonzero_value_flushes_to_zero(self):
        # Round-to-nearest maps entries below half an LSB to exactly zero;
        # for second moments that yields v=0 -> denominator ~eps -> blow-up.
        opt = SMO8bit([torch.nn.Parameter(torch.zeros(32, 32))])
        torch.manual_seed(19)
        data = torch.cat([torch.randn(8, 64) * 100.0, torch.randn(56, 64) * 1e-3])
        q, s, shape = opt._quantize_blockwise(data, 64)
        deq = opt._dequantize_blockwise(q, s, shape)
        flushed = ((deq == 0) & (data != 0)).sum().item()
        self.assertEqual(flushed, 0)
        # and the reconstruction error stays within one LSB of the block scale
        block_max = data.abs().view(-1, 64).max(dim=1).values.repeat_interleave(64)
        self.assertTrue((deq.flatten() - data.flatten()).abs().max().item()
                        <= (block_max / 127.0).max().item() * 1.5)

    def test_conv_states_use_row_aligned_blocks_linear_keeps_flat(self):
        torch.manual_seed(3)
        conv = torch.nn.Parameter(torch.randn(64, 8, 3, 3))
        linear = torch.nn.Parameter(torch.randn(64, 72))
        opt_conv = SMO8bit([conv], lr=1e-3, k_ratio=0.25, block_size=64, compress_conv=True)
        opt_lin = SMO8bit([linear], lr=1e-3, k_ratio=0.25, block_size=64)
        conv.grad = torch.randn_like(conv)
        linear.grad = torch.randn_like(linear)
        opt_conv.step()
        opt_lin.step()

        # comp view (64, 72) -> comp shape (16, 18) -> one int8 scale per
        # compact row (block = min(block_size, comp_w))
        self.assertEqual(opt_conv.state[conv]["q_block"], 18)
        # linear keeps the historical flattened-block layout
        self.assertEqual(opt_lin.state[linear]["q_block"], 64)

    def test_row_aligned_blocks_do_not_crush_low_scale_rows(self):
        opt = SMO8bit([torch.nn.Parameter(torch.zeros(32, 32))])
        torch.manual_seed(17)
        big, small = 100.0, 0.01
        data = torch.cat([torch.randn(1, 64) * big, torch.randn(1, 64) * small], dim=0)

        def max_err_on_small_row(block):
            q, s, shape = opt._quantize_blockwise(data, block)
            deq = opt._dequantize_blockwise(q, s, shape)
            return (deq[1] - data[1]).abs().max().item()

        flat_err = max_err_on_small_row(128)
        row_err = max_err_on_small_row(64)

        # Flat blocks share one scale dominated by the big row (~big/254);
        # row-aligned error must sit at the small row's own resolution.
        self.assertLess(row_err, small * 4 / 127)
        self.assertGreater(flat_err, row_err * 10)


class LowPeakConvTests(unittest.TestCase):
    def test_banded_matches_monolithic_on_conv_shape(self):
        # (128, 8, 4, 4) -> row view (128, 128); exact pool at k=0.25 -> (32, 32)
        torch.manual_seed(11)
        init = torch.randn(128, 8, 4, 4)

        param_mono = torch.nn.Parameter(init.clone())
        param_band = torch.nn.Parameter(init.clone())
        opt_mono = SMO8bit([param_mono], lr=1e-3, k_ratio=0.25, block_size=64, compress_conv=True)
        opt_band = SMO8bit([param_band], lr=1e-3, k_ratio=0.25, block_size=64,
                           low_peak=True, band_mb=0.05, compress_conv=True)

        torch.manual_seed(23)
        for _ in range(3):
            grad = torch.randn_like(init)
            param_mono.grad = grad.clone()
            param_band.grad = grad.clone()
            opt_mono.step()
            opt_band.step()

        state_mono = opt_mono.state[param_mono]
        state_band = opt_band.state[param_band]
        for key in ("m_q", "v_q"):
            self.assertTrue(torch.equal(state_mono[key], state_band[key]))
        for key in ("m_s", "v_s"):
            self.assertTrue(torch.allclose(state_mono[key], state_band[key]))
        self.assertTrue(torch.allclose(param_mono.view(128, 128), param_band.view(128, 128), atol=1e-5))



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
