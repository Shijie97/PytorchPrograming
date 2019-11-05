# !user/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for s in size:
            num *= s
        return num

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )

params = list(net.parameters())

print(len(params)) # 10

print(params[0])

# Parameter containing:
# tensor([[[[ 0.0763,  0.1148,  0.0531, -0.0528, -0.1421],
#           [ 0.0018, -0.0478,  0.0947, -0.1419,  0.1277],
#           [-0.1875, -0.1713, -0.1803,  0.0554,  0.1521],
#           [-0.0135,  0.0773,  0.1243, -0.1398,  0.1202],
#           [-0.0042, -0.0470, -0.0779, -0.1225,  0.1758]]],
#
#
#         [[[ 0.0604,  0.0165, -0.1500,  0.1268, -0.1183],
#           [-0.1449, -0.1218, -0.0770,  0.1751, -0.1787],
#           [ 0.1886,  0.0515,  0.1493,  0.1791, -0.0915],
#           [-0.1003, -0.0489, -0.1839, -0.1314,  0.1498],
#           [-0.0817,  0.1188, -0.0541, -0.1719,  0.0217]]],
#
#
#         [[[ 0.1395, -0.0934, -0.0700, -0.1377,  0.1531],
#           [-0.0849, -0.0615, -0.1036, -0.0591,  0.0860],
#           [-0.1454,  0.1685, -0.1769, -0.0706, -0.0359],
#           [-0.0047, -0.0379, -0.0459, -0.1072,  0.0666],
#           [ 0.1885, -0.1255,  0.0625,  0.0603, -0.0201]]],
#
#
#         [[[-0.0339,  0.0782,  0.1737, -0.0216, -0.0364],
#           [ 0.0502,  0.1786,  0.0379, -0.0154,  0.0841],
#           [-0.0485,  0.0332,  0.1251, -0.1607, -0.0169],
#           [-0.1867,  0.1518, -0.1023,  0.1439,  0.0592],
#           [-0.0921, -0.0311,  0.0058, -0.1839, -0.1826]]],
#
#
#         [[[-0.0854, -0.1493,  0.0130, -0.1636, -0.0028],
#           [ 0.0872, -0.1662,  0.1428,  0.0675, -0.0140],
#           [ 0.0015,  0.1216,  0.1871,  0.0931,  0.0794],
#           [-0.0473, -0.1567,  0.1145, -0.1586, -0.1121],
#           [ 0.1730,  0.1142, -0.0020, -0.0508, -0.1951]]],
#
#
#         [[[-0.1987, -0.1774,  0.1361, -0.1865,  0.1565],
#           [ 0.0636,  0.1042, -0.0114,  0.0809,  0.1862],
#           [-0.1830, -0.0908, -0.0694, -0.0396, -0.0330],
#           [ 0.1118,  0.0124,  0.0951, -0.1285,  0.0293],
#           [-0.1882, -0.1337, -0.1575,  0.1629, -0.1629]]]], requires_grad=True)

# 下面我们来看一下net中到底保存着哪些参数
for i in range(len(params)):
    print(str(i))
    print(params[i].size())
# 0 第一个卷积层的卷积核参数
# torch.Size([6, 1, 5, 5])
# 1 第一个卷积层的偏置参数
# torch.Size([6])
# 2 第二个卷积层的卷积核参数
# torch.Size([16, 6, 5, 5])
# 3 第二个卷积层的偏置参数
# torch.Size([16])
# 4 第一个全连接层的权重参数，下同，注意，这里的(120， 400)指的是输入参数个数是400，输出是120，这里严格按照U^t * x来的，x是列向量，因此这里U为转置
# torch.Size([120, 400])
# 5 第一个全连接层的偏置参数，下同
# torch.Size([120])
# 6
# torch.Size([84, 120])
# 7
# torch.Size([84])
# 8
# torch.Size([10, 84])
# 9
# torch.Size([10])

# 随机产生一个输入
x = torch.randn(1, 1, 32, 32) # 输入是(N, C, H, W)
out = net(x)
print(out)
# tensor([[ 0.0131, -0.0863, -0.0145,  0.0378, -0.0305,  0.0052,  0.0071, -0.0702,
#           0.0249, -0.0091]], grad_fn=<AddmmBackward>)

# 反向传播时，先用zero_grad将所有权重的梯度缓存置0
# 我想这里应该不是把权重置0，因为权重不能置0！

# 注意，反向传播的时候，输入的必须是1 * 10的矩阵，因为最后的输出的shape是(N, 10)
# 具体原因，官网Linear有详细解答，因为输入的时候为y = x * A^t + b，则x应该是(N, 行向量的形式)
# 所以呢，输出也应该是一个行向量哇~

# out.backward(torch.randn(1, 10))
# tensor([[ 0.0784, -0.0498, -0.0177,  0.0918,  0.1104,  0.0504,  0.1579, -0.1149,
#          -0.0029, -0.0335]], grad_fn=<AddmmBackward>)

# 有了输出结果，我想计算误差肿么办呢？
# 这里，我们先用均方误差来试一下吧！
target = torch.rand(1, 10)
MSE = nn.MSELoss()
loss = MSE(out, target) # 体会到了函数式编程~
print(loss) # tensor(0.2491, grad_fn=<MseLossBackward>)

# 反向传播的过程中，所有require_grad为True的张量都会累计梯度
# 下面，我们来看一下传播路径

print(loss.grad_fn) # <MseLossBackward object at 0x0000021479030F28> grad_fn指向上一个（即反向的下一个）object

# 下面，我们基于loss进行反向传播

# 在反向传播之前，我们看一下fc1层的权重矩阵
net.zero_grad()
print(net.fc1.weight)
# tensor([[-0.0443,  0.0075, -0.0183,  ..., -0.0023, -0.0213, -0.0253],
#         [ 0.0038, -0.0305, -0.0318,  ..., -0.0458,  0.0053, -0.0357],
#         [ 0.0372, -0.0385, -0.0340,  ..., -0.0099, -0.0498, -0.0357],
#         ...,
#         [-0.0199, -0.0323, -0.0132,  ..., -0.0398,  0.0402,  0.0098],
#         [-0.0100,  0.0211, -0.0425,  ...,  0.0433, -0.0103, -0.0339],
#         [ 0.0122,  0.0241,  0.0354,  ..., -0.0219,  0.0001, -0.0240]],
#        requires_grad=True)
print(net.fc1.weight.grad) # None，最开始的梯度是不存在的，所以是None

# 反向传播之后，再看一下咯
loss.backward()
print(net.fc1.weight.grad)
# tensor([[-0.0443,  0.0075, -0.0183,  ..., -0.0023, -0.0213, -0.0253],
#         [ 0.0038, -0.0305, -0.0318,  ..., -0.0458,  0.0053, -0.0357],
#         [ 0.0372, -0.0385, -0.0340,  ..., -0.0099, -0.0498, -0.0357],
#         ...,
#         [-0.0199, -0.0323, -0.0132,  ..., -0.0398,  0.0402,  0.0098],
#         [-0.0100,  0.0211, -0.0425,  ...,  0.0433, -0.0103, -0.0339],
#         [ 0.0122,  0.0241,  0.0354,  ..., -0.0219,  0.0001, -0.0240]],
#        requires_grad=True)