import torch

def create_hadamard_matrix(N, dtype, device):
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    n = 1
    while n < N:
        H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)
        n *= 2
    return H

def fwht_1d(x):
    N = x.shape[-1]
    h = 1
    x = x.clone()
    while h < N:
        shape = x.shape
        x_reshaped = x.view(*shape[:-1], N // (2 * h), 2, h)
        x_0 = x_reshaped[..., 0, :].clone()
        x_1 = x_reshaped[..., 1, :].clone()
        x_reshaped[..., 0, :] = x_0 + x_1
        x_reshaped[..., 1, :] = x_0 - x_1
        h *= 2
    return x

def fwht_2d(x):
    return fwht_1d(fwht_1d(x).t()).t()

x = torch.randn(8, 8)
H8 = create_hadamard_matrix(8, x.dtype, x.device)

# Matrix mult 2D
res_mat = H8 @ x @ H8.t()

# Fast WHT 2D
res_fast = fwht_2d(x)

print("Max diff:", (res_mat - res_fast).abs().max().item())
