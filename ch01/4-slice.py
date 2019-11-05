# !user/bin/python
# -*- coding: UTF-8 -*-

import torch
import numpy as np

# pytorch也可以有切片

x = torch.arange(10)
print(x)

print(x[2 : -2]) # tensor([2, 3, 4, 5, 6, 7])