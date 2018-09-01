#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/8/28 14:38
# @Author : ly
# @File   : pytorch6.py
import torch
import torch.utils.data as Data

torch.manual_seed(0)  # reproducible

BATCH_SIZE = 5  # 批训练的数据个数

x = torch.linspace(1, 10, 10)  # x data (torch tensor)
y = torch.linspace(10, 1, 10)  # y data (torch tensor)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(x, y)
# 把 dataset 放入 DataLoader
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

if __name__ == '__main__':
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())
