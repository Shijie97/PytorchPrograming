# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

x = torch.empty(5, 3)
print(x)
# tensor([[1.0286e-38, 8.4490e-39, 6.9796e-39],
#         [1.0561e-38, 8.7245e-39, 1.0745e-38],
#         [1.0653e-38, 1.0286e-38, 1.0194e-38],
#         [9.2755e-39, 1.0561e-38, 1.0102e-38],
#         [4.2246e-39, 1.1112e-38, 1.4013e-43]])
y = torch.zeros(5, 3)
print(y)

# tensor([[1.0286e-38, 8.4490e-39, 6.9796e-39],
#         [1.0561e-38, 8.7245e-39, 1.0745e-38],
#         [1.0653e-38, 1.0286e-38, 1.0194e-38],
#         [9.2755e-39, 1.0561e-38, 1.0102e-38],
#         [4.2246e-39, 1.1112e-38, 1.4013e-43]])

x = torch.tensor([5.5, 3], dtype = torch.double)
print(x)
# tensor([5.5000, 3.0000], dtype=torch.float64)

# x.new_ones生成的张量具有和x相同的属性
y = x.new_ones(5, 5)
print(y)
# tensor([[1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1.]], dtype=torch.float64)

# 随机张量的shape和x一样
z = torch.rand_like(x)
print(z)
# tensor([0.1793, 0.6121], dtype=torch.float64)

