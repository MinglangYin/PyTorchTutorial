"""
Author: Minglang Yin
    Calculate gradient of polynomial by .backward()
"""

import torch
import torch.optim as optim

torch.manual_seed(123456)

def poly(x):
    return x[0]**2 + 5*x[1] + 2 

def poly_2(x):
    return x*3 + 2

x_array = [2.0, 3.0]
x = torch.tensor(x_array, requires_grad=True)
optimizer = optim.Adam([x], lr = 0.01)

optimizer.zero_grad()
y = poly(x)
y.backward(retain_graph=True)
print('x grad is ',x.grad)
optimizer.zero_grad()
y2 = poly_2(y)
y2.backward()
print('x grad 2 is ', x.grad)