#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/8/27 21:07
# @Author : ly
# @File   : pytorch3.py
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 将一维的数据变成二维
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# print(x)

y = x.pow(2) + 0.2 * torch.rand(x.size())
# print(0.2 * torch.rand(x.size()))

x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
# print(net)  # net 的结构
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

plt.ion()   # 画图
plt.show()
for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

# 保存神经网络
# torch.save(net, "net.pkl")
# 保存神经网络中的参数
# torch.save(net.state_dict(), "net_params.pkl")

print(net.state_dict())

# 提取神网络
# net1 = torch.load("net.pkl")
# print(net1(x))

# 构建神经网络，将参数读入，并给与神经网络
# net2 = Net(n_feature=1, n_hidden=10, n_output=1)
# net2.load_state_dict(torch.load("net_params.pkl"))
# print(net2(x))
