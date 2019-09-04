"""
Author: Minglang Yin
    Examplify the routine to calculate output gradient w.r.t. input 
    from an Neural Network. 

Note: Gradients do not accumulate over times
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

torch.manual_seed(123456)
np.random.seed(0)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.layer(x)

goal = torch.Tensor([[1.0],[1.0]])
x_array = [[2.0],[5.0]]
x = torch.tensor(x_array, requires_grad=True)

net = model()
optimizer = optim.Adam(net.parameters(), lr = 0.01)

for i in range(0, 100):   
    optimizer.zero_grad()
    y = net(x)
    
    # gradient does not accumulate after repetitively calculating gradient
    tmp = torch.ones(2, 1, dtype=torch.float32)
    x_grad = torch.autograd.grad(y, x, tmp, retain_graph=True)
    print('1. x_gradient = ', x_grad)
    x_grad = torch.autograd.grad(y, x, tmp, retain_graph=True)
    print('2. x_gradient = ', x_grad)

    loss = ((y - goal)**2).mean()
    loss.backward(x)
    print('backward:', x.grad)
    optimizer.step()

print('network prediction:', net(x))