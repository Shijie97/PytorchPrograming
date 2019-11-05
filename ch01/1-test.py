# !user/bin/python
# -*- coding: UTF-8 -*-

import torch
from torch.backends import cudnn
import torch.nn.functional as F

a = torch.tensor(1.)
# 若正常则静默

print(a.cuda())
# 若正常则返回 tensor(1., device='cuda:0')

print(cudnn.is_available())
# 若正常则返回 True

print(cudnn.is_acceptable(a.cuda()))
# 若正常则返回 True

print(torch.__version__) # 1.2.0

print(torch.cuda.is_available()) # True

x = torch.Tensor([4.0] * 4).view(1, 4)
print(F.softmax(x, dim = 1)) # tensor([[0.2500, 0.2500, 0.2500, 0.2500]])

print(torch.cuda.device_count()) # 1

print(torch.cuda.get_device_name(0)) # GeForce GTX 960M

print(torch.cuda.current_device()) # 0