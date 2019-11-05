# !user/bin/python
# -*- coding: UTF-8 -*-

import torch

print(torch.cuda.is_available())

device = torch.device('cuda', 0)
x = torch.arange(1000000).view(1000, 1000)
y = torch.ones_like(x, device = device)
x = x.to(device)
z = x + y
print(z)