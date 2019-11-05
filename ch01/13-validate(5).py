# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

# unsqueeze详解
# unsqueeze不改变原有数据

x = torch.tensor([1, 2, 3, 4])
print(x.unsqueeze(0))
# print(x.unsqueeze(0))
print(x.unsqueeze(1))
# tensor([[1],
#         [2],
#         [3],
#         [4]])