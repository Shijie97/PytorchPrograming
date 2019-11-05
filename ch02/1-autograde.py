# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

# 定义一个x，x是需要求梯度的
x = torch.ones(2, 2, requires_grad = True)

# y在x上面加2
y = x + 2
print(y) # 发现y的grad_fn属性是指向A的
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)

z = torch.empty_like(y, requires_grad = True)
z = y * y * 3
out = z.mean()
# print(z)
# z.backward()

# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward0>)
print(out) # out现在仍然持有着x对象
# tensor(27., grad_fn=<MeanBackward0>)

# out是一个标量，所以直接用out.backward进行反向传播，相当于dout = 1
out.backward()

# 反向传播完之后，求grad
# 注意，只有变量的requires_grad为True，才能求它的grad，它的持有者并不能求
print(x.grad)
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])
print(y.grad) # None
print(z.grad) # None