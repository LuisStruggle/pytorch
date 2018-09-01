#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/8/27 19:30
# @Author : ly
# @File   : pytorch1.py
import torch
import numpy as np
from torch.autograd import Variable

np_data = np.arange(6).reshape((2, 3))
np_datatorch = torch.from_numpy(np_data)
np_datanew = np_datatorch.numpy()

# print(np_datatorch)
# print(np_datanew)

data = torch.FloatTensor([-1, -2, 3, -4])
# print(data)

newdata = torch.abs(data).reshape((2, 2))
# print(newdata)

mm = torch.mm(newdata, newdata)
# print(mm)

data = torch.FloatTensor([[1, 2], [3, 4]])
var = Variable(data, requires_grad=True)
# print(var)

data1 = torch.mean(data)
data2 = torch.mean(var)
# print(data1, data2)

print(var.data.numpy())

