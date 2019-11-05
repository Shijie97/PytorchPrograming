# !user/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# 利用Compose，对所有预处理操作进行汇总
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)])
# 构造训练集
trainset = datasets.CIFAR10(root = './cifar10', train = True, download = True, transform = transform)
# print('len(trainset) = ', len(trainset)) # len(trainset) =  50000
# # trainset可以看做一个数组，每个数组里面的元素包含两个部分，一个是三维的数据，一个是对应的标签
# print(trainset[0][0].size(), trainset[0][1]) # torch.Size([3, 32, 32]) 6


trainloader = DataLoader(dataset = trainset, batch_size = 50, shuffle = True, num_workers = 0)


# print('len(trainloader) = ', len(trainloader)) # len(trainloader) =  100
# print(list(trainloader)[0][0].size()) # torch.Size([4, 3, 32, 32])
# print(list(trainloader)[0][1]) # tensor([7, 6, 1, 8])


# 构造测试集
testset = datasets.CIFAR10(root = './cifar10', train = False, download = True, transform = transform)
# 用dataloader对象将测试集装起来
testloader = DataLoader(dataset = testset, batch_size = 50, shuffle = True, num_workers = 0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 下面来定义网络模型

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)  # pool可以复用，写一个就OK
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 30, 5)
        self.conv3 = nn.Conv2d(30, 60, 3)
        self.fc1 = nn.Linear(60 * 3 * 3, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 60 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device = ', device)
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params = net.parameters(), lr = 0.01, momentum = 0.9)


# 训练数据
for epoch in range(10): # 这里设置epoch为20
    loss_temp = 0.0
    correct_num = 0
    for data in trainloader:
        # 获取输入样本和标签
        # 这里的input为(50, 3, 32, 32)， labels为(50,)
        # inputs, labels = data # 这么看来data应该是个元组了
        inputs, labels = data[0].to(device), data[1].to(device)

        # print('inputs.shape', inputs.shape) # inputs.shape torch.Size([50, 3, 32, 32])
        # print('labels.shape', labels.shape) # labels.shape torch.Size([50])

        # 每次反向传播是，梯度都要被重置
        optimizer.zero_grad()
        # 前向
        outputs = net(inputs)
        # print('outputs.shape', outputs.shape) # outputs.shape torch.Size([50, 10])

        # 求损失
        loss = criterion(outputs, labels)
        # 反向
        loss.backward()
        # 更新梯度
        optimizer.step()

        loss_temp += loss.item()
        correct_num_per_round = torch.sum(torch.argmax(outputs, dim = 1) == labels).item()
        correct_num += correct_num_per_round

    loss_per_epoch = round(loss_temp / 1000, 4) # 循环1000次才能遍历全部数据
    acc = round(correct_num / 50000, 4)
    print('epoch = ' + str(epoch) + ', loss = ' + str(loss_per_epoch) + ', acc = ' + str(acc))


print('trainning finished')

# 开始进行测试
print('#' * 20)
print('test begin!')
correct_num_each_class = np.zeros(10)
total_num_each_class = np.zeros(10)
correct_num_total = 0

with torch.no_grad():
    for i, data in enumerate(testloader):
        # print('current index is', i) # i从0 ~ 199
        inpupts, labels = data[0].to(device), data[1].to(device)
        outputs = net(inpupts)
        outputs = torch.argmax(outputs, dim = 1)
        for j in range(outputs.size()[0]):
            total_num_each_class[labels[j]] = total_num_each_class[labels[j]] + 1
            if outputs[j] == labels[j]:
                correct_num_each_class[labels[j]] = correct_num_each_class[labels[j]] + 1
                correct_num_total = correct_num_total + 1

    print('test acc is', round(correct_num_total / 10000, 4))
    for i in range(10):
        acc = correct_num_each_class[i] / total_num_each_class[i]
        print(classes[i] + '\'s acc is ' + str(round(acc, 4)))

print('test finish!')

