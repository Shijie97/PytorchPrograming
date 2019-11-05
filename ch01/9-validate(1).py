# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

# 验证一下两个张量比较生成布尔张量的过程

a = torch.arange(16).view(4, 4)
b = torch.argmax(a, dim = 1)
print(b)
print([round(x.item(), 5) for x in b])

z = torch.tensor([3, 1, 2, 5], dtype = torch.long) # 类型必须保持一致
z = z.view(-1, 1)
b = b.view(-1, 1)
print(b)
print(z)
print(b == z)
# tensor([[ True],
#         [False],
#         [False],
#         [False]])
print(torch.sum(b == z)) # tensor(1)