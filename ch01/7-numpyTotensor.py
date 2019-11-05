# !user/bin/python
# -*- coding: UTF-8 -*-

import torch
import numpy as np

# numpy转tensor，用torch.Tensor
x = np.arange(10)
y = torch.Tensor(x)

x = x + 1
print(x) # [ 1  2  3  4  5  6  7  8  9 10]
print(y) # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

# tensor转numpy
x = torch.arange(10)
y = x.numpy()
x = x + 1
print(x) # tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
print(y) # [0 1 2 3 4 5 6 7 8 9]
y = y + 10
print(x) # tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
print(y) # [10 11 12 13 14 15 16 17 18 19]