#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/8/28 12:14
# @Author : ly
# @File   : pytorch5.py
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)

print(net)
