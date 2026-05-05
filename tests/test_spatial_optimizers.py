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


if __name__ == "__main__":
    unittest.main()
