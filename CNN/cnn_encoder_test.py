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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = out2.reshape(out2.size(0), -1)
        out = self.fc(out3)
        return out

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def main():
    ## Hyper parameters
    learning_rate = 0.001
    cuda = 1
    epochs = 1000
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    
    ## pre-proc data
    input_size = [1, 1, 16, 16]
    output_size = [10]
    input_tensor = torch.randn(input_size).to(device)
    output_tensor = torch.randn(output_size).to(device)

    ## model
    model = ConvNet().to(device)
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

if __name__=='__main__':
    main()





