import torch
from optim_walsh_pure import fwht_2d, ifwht_2d

x = torch.arange(16, dtype=torch.float32).view(4, 4)
print("x:\n", x)

x_f = fwht_2d(x)
print("\nx_f:\n", x_f)

x_f_trunc = torch.zeros_like(x_f)
x_f_trunc[:2, :2] = x_f[:2, :2]

x_rec = ifwht_2d(x_f_trunc)
print("\nx_rec:\n", x_rec)

from test_dct import create_dct_matrix
from optim_dct_pure import dct_2d, idct_2d

D = create_dct_matrix(4, torch.float32, 'cpu')
x_dct = dct_2d(x, D, D)
x_dct_trunc = torch.zeros_like(x_dct)
x_dct_trunc[:2, :2] = x_dct[:2, :2]
x_rec_dct = idct_2d(x_dct_trunc, D, D)

print("\nx_rec_dct:\n", x_rec_dct)
print("\nMin of x_rec_dct:", x_rec_dct.min().item())

