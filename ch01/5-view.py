# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

# 姑且当做reshape

x = torch.arange(28)
print(x.view(4, 7))
# tensor([[ 0,  1,  2,  3,  4,  5,  6],
#         [ 7,  8,  9, 10, 11, 12, 13],
#         [14, 15, 16, 17, 18, 19, 20],
#         [21, 22, 23, 24, 25, 26, 27]])

# 不改变原有值
print(x.size())
# torch.Size([28])