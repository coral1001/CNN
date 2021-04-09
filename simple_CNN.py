# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 20:46:47 2021

@author: qg
"""

import torch
in_channels, out_channels = 5, 10
width,height = 100,100 #图像大小
kernel_size = 3 #卷积核大小3*3
batch_size = 1 

input = torch.randn(batch_size, in_channels,width,height)

conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)#输出三个数值输出通道，输入通道，卷积核大小