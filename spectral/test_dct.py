import torch
import math

def create_dct_matrix(N, dtype, device):
    n = torch.arange(N, dtype=dtype, device=device)
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
    dct_mat = torch.cos(math.pi / N * (n + 0.5) * k)
    dct_mat[0] *= 1.0 / math.sqrt(2.0)
    dct_mat *= math.sqrt(2.0 / N)
    return dct_mat

D = create_dct_matrix(8, torch.float32, 'cpu')
print("Orthogonality check (D^T @ D):")
print(torch.round(D.t() @ D * 1e4) / 1e4)

x = torch.randn(8, 8)
x_dct = D @ x @ D.t()

x_rec = D.t() @ x_dct @ D

print("\nReconstruction max diff:", (x - x_rec).abs().max().item())

print("\nEnergy of x vs x_dct:")
print(x.pow(2).sum().item(), "vs", x_dct.pow(2).sum().item())
