import torch
import torch.nn.functional as F

import numpy as np

a = torch.ones((2, 3, 28, 28))
b = torch.rand((2, 3, 28, 28))

print(F.binary_cross_entropy(a, a, reduction="mean"))
print(F.binary_cross_entropy(b, b, reduction="mean"))

print(F.binary_cross_entropy_with_logits(b, b))

print(torch.nn.L1Loss()(a, b))
print(torch.nn.MSELoss()(b, b))
print(torch.sum(torch.abs(a - b)) / (28 * 28 * 3 * 2))

mu = torch.zeros((10, 1))
logvar = torch.rand((10, 1))
torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

torch.abs(a - b).numpy().shape

torch.mean(torch.sum(torch.abs(a - b), dim=1), dim=0)

d = np.ones((2, 32, 32, 3))
c = np.ones((2, 10, 10, 3))
np.sum(c, axis=0)
