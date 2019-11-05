# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

x = torch.tensor([[2, 2],
                  [3, 3]])
y = torch.tensor([[1, 1],
                  [4, 4]])

# 三种方法
res1 = x + y
res2 = torch.add(x, y)
res3 = torch.empty_like(x)
torch.add(x, y, out = res3)

# add_会改变y的值
y.add_(x)

print(res1)
print(res2)
print(res3)
# tensor([[3, 3],
#         [7, 7]])
print(y)
# tensor([[3, 3],
#         [7, 7]])

# 方法名后面带有下标的，都会改变自身，比如下面这个转置
x.t_()
print(x)
# tensor([[2, 3],
#         [2, 3]])