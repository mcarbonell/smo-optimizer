"""Quick smoke checks for SMO-Spatial and SMO-Spatial-8bit consistency."""

import torch

from smo import SMO, SMO8bit


# Benchmark classification: family=optimizer_step, category=smoke, status=canonical


def main():
    torch.manual_seed(1234)
    base = torch.randn(32, 32, dtype=torch.float32)
    grad = torch.randn_like(base)

    p_spatial = torch.nn.Parameter(base.clone())
    p_8bit = torch.nn.Parameter(base.clone())

    opt_spatial = SMO([p_spatial], lr=1e-3, k_ratio=0.25)
    opt_8bit = SMO8bit([p_8bit], lr=1e-3, k_ratio=0.25, block_size=1)

    p_spatial.grad = grad.clone()
    p_8bit.grad = grad.clone()

    opt_spatial.step()
    opt_8bit.step()

    state_spatial = opt_spatial.state[p_spatial]
    state_8bit = opt_8bit.state[p_8bit]
    v_8bit = opt_8bit._dequantize_blockwise(state_8bit["v_q"], state_8bit["v_s"], state_8bit["comp_shape"])

    print("SMOKE: spatial vs 8-bit consistency")
    print(f"max |param delta|: {(p_spatial - p_8bit).abs().max().item():.8f}")
    print(f"max |v delta|:     {(state_spatial['exp_avg_sq'] - v_8bit).abs().max().item():.8f}")


if __name__ == "__main__":
    main()
