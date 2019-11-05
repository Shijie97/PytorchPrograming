# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

# 通过len返回tensor的个数，不太好
# 我们用tensor.size()[0]
x = torch.tensor([[1, 2, 3], [1, 2, 3]])
print(x.size()[0])