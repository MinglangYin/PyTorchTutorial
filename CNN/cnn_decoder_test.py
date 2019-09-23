"""
Minglang Yin, Brown University
minglang_yin@brown.edu

Decoder test
"""
import sys
sys.path.insert(0, '../')

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
import time
import torch.nn as nn

from Utils.others import to_numpy
from Models.encoder import Encoder
from Utils.parse import parse
# from torchsummary import summary


# Hyper parameters
learning_rate = 0.001
cuda = 1
epochs = 500

blk_1 = [
    ['conv2d', 1, 3],
    ['batchnorm2d'],
    ['relu'],
    ['UpsamplingNearest2d', 'scale_factor=2']
]
blk_2 = [
    ['conv2d', 1, 3],
    ['batchnorm2d'],
    ['relu'],
    ['UpsamplingNearest2d', 'scale_factor=2']
]
blk_4 = [
    ['reshape', 'shape=-1'],
    ['linear', 10],
    # ['softmax']
]

input_size = [1, 1, 5, 5]
output_size = [10]

parser = parse()
blks = [blk_1, blk_2, blk_4]

## input and output size (assume square image now)
################## tonight I need to add more filter channels into parsing part, and also need to think about the way to do it for PINNs

parser = parse()
net_struct = parser.get_struct(blks, input_size, output_size)

## environment
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

## load data


## model
model = Encoder(net_struct, input_size, output_size).to(device)

### bug!! only works for single GPU
# summary(model, input_size = (1, 1, 28))

##
input_tensor = torch.randn(1, 1, 5, 5).to(device)
output_tensor = torch.randn(10).to(device)

## model
model = Encoder(net_struct, input_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(epoch):
    model.train()
    def closure():
        optimizer.zero_grad()
        output_pred = model(input_tensor)
        loss = ((output_pred - output_tensor)**2).mean()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    loss_value = loss.item() if not isinstance(loss, float) else loss
    print(f'epoch {epoch}: loss {loss_value:.6f}')

print('start training...')
tic = time.time()
for epoch in range(1, epochs + 1):
    train(epoch)

output_pred = model(input_tensor)
print(f'output_pred: {to_numpy(output_pred)}')
print(f'output_train: {to_numpy(output_tensor)}')







