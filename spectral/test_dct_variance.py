import torch
from test_dct import create_dct_matrix
from optim_dct_pure import dct_2d, idct_2d

# Create a "gradient" with some zero variance in some places to see if variance goes negative
g = torch.randn(8, 8) * 10
g[2:6, 2:6] = 0.0 # Some zeros

g_sq = g ** 2

D = create_dct_matrix(8, torch.float32, 'cpu')
g_sq_dct = dct_2d(g_sq, D, D)

# Truncate
g_sq_dct_trunc = torch.zeros_like(g_sq_dct)
g_sq_dct_trunc[:4, :4] = g_sq_dct[:4, :4]

# Reconstruct
g_sq_rec = idct_2d(g_sq_dct_trunc, D, D)

print("Original min of g^2:", g_sq.min().item())
print("Reconstructed min of g^2:", g_sq_rec.min().item())
print("Reconstructed max of g^2:", g_sq_rec.max().item())
print("Negative values in reconstructed g_sq?", (g_sq_rec < 0).sum().item())

