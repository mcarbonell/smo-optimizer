"""Quick smoke checks for SMO-Spatial and SMO-Spatial-8bit consistency."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks._paths import add_project_root_to_path
add_project_root_to_path()

from smo import SMO, SMO8bit


def set_seed(seed: int):
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Smoke check: SMO vs SMO8bit consistency")
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    
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

    print("SMOKE: spatial vs 8-bit consistency (seed={})".format(args.seed))
    print(f"max |param delta|: {(p_spatial - p_8bit).abs().max().item():.8f}")
    print(f"max |v delta|:     {(state_spatial['exp_avg_sq'] - v_8bit).abs().max().item():.8f}")
    
    # Success criterion
    max_param_delta = (p_spatial - p_8bit).abs().max().item()
    max_v_delta = (state_spatial['exp_avg_sq'] - v_8bit).abs().max().item()
    if max_param_delta < 1e-5 and max_v_delta < 1e-5:
        print("[OK] CONSISTENT")
        return 0
    else:
        print("[FAIL] MISMATCH")
        return 1


if __name__ == "__main__":
    exit(main())