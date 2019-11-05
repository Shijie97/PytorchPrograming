# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

x = torch.arange(16).view(4, 4)
print(x)

# print(x.item()) ERROR!

# item用于获取张量中某个位置的那个数，只返回那个数
# 如果不加item，返回的将是tensor(XXX)
# 不能同时返回多个位置，只能返回一个数
print(x[0, 0].item()) # 0
print(x[0, 0]) # tensor(0)
