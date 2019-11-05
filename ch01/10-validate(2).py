# !user/bin/python
# -*- coding: UTF-8 -*-

# 验证一下交叉熵函数
# 这个交叉熵函数包含了softmax和求交叉熵
# label标签不用是one-hot，一个数即可

import torch
from torch.nn.functional import cross_entropy
from torch.nn.functional import softmax

x = torch.tensor([0.3, 0.6, 0.1]).view(1, 3) # 需要是一个二维矩阵
y = torch.tensor([1]) # 不能是二维矩阵，只需要是一个行向量即可，向量长度为上面二维矩阵的行数
res = cross_entropy(x, y)
print(res) # tensor(0.8533)

x = softmax(x, dim = 1)
print(x) # tensor([[0.3156, 0.4260, 0.2584]])
# print(-torch.log(torch.tensor(x[0, 1])))
print(x[0, 1]) # tensor(0.4260)
z = x[0, 1]
print(-torch.log(z)) # tensor(0.8533) 和上面一样