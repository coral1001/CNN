# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:08:25 2021

@author: qg
"""

import torch

input = [3,4,5,7,
         2,3,6,7,
         1,3,5,7,
         7,4,7,8]
input= torch.Tensor(input).view(1,1,4,4,)#分别是batch_size,channel,width,height

#conv_layer = torch.nn.Conv2d(1, 1, kernel_size = 3,padding =1,bias= False)#stride为步长
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
#kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)#分别是输出通道数，输入通道数，卷积核大小

#conv_layer.weight.data = kernel.data

output = maxpooling_layer(input)
print(output)