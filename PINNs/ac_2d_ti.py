"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving Allen-Cahn equation (2D) at 
        equilibrium state.
    
    grad*(grad(phi)) + exp(-x)*(x-2+y^3+6*y) = 0
    with
        phi(0, y) = y^3, 
        phi(1, y) = (1+y^3)*exp(-1)
        phi(x, 0) = x*exp(-x)
        phi(x, 1) = exp(-x)*(x+1)

    where:
        b/del= 10 = D

    Analytical solution: phi = 0.5(1-tanh(b/del*x))

    Input: [x]
    Output: [u]
"""

import sys
sys.path.insert(0,'../Utils')

import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plotting import newfig, savefig

torch.manual_seed(123456)
np.random.seed(123456)

eta = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(2, 10))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for num in range(2,7):
            self.net.add_module('linear_layer_%d' %(num+1), nn.Linear(10, 10))
            self.net.add_module('tanh_layer_%d' %(num+1), nn.Tanh())
        self.net.add_module('linear_layer_6', nn.Linear(10, 1))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        u = self.net(x)
        u_g = gradients(u, x)[0]
        u_x, u_y = u_g[:, 0], u_g[:, 1]
        u_xx = gradients(u_x, x)[0][:, 0]
        u_yy = gradients(u_y, x)[0][:, 1]

        X, Y = x[:, 0], x[:, 1]
        loss = (u_xx + u_yy) - torch.exp(-X)*(X - 2 + Y**3 + 6*Y) ### solving PDE with NN: equ 8, 9
        return (loss**2).mean()

    def loss_bc(self, x, u):
        return ((self.net(x) - u)**2).mean()

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def AC_2D_solu(x):
    return np.exp(-x[:,0])*(x[:,0]+x[:,1]**3)

def main():
    ## parameters
    device = torch.device(f"cpu")
    epochs = 10100
    num_b_train = 10
    num_f_train = 50
    num_test = 50
    lr = 0.001

    x = np.linspace(0, 1, num=num_test)
    y = np.linspace(0, 1, num=num_test)
    x_grid, y_grid = np.meshgrid(x, y)
    x_test = np.concatenate((x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    
    x_up = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_dw = np.vstack((x_grid[0,:], y_grid[0,:])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_b = np.vstack((x_up, x_dw, x_l, x_r))

    u_b = AC_2D_solu(x_b).reshape(num_f_train*4, 1)
    u_f = AC_2D_solu(x_test).reshape(num_f_train*num_f_train, 1)

    id_f = np.random.choice(num_test*num_test, num_f_train)
    id_b = np.random.choice(num_test*4, num_b_train)

    x_f_train = torch.tensor(x_test[id_f], requires_grad=True, dtype=torch.float32).to(device)
    x_b_train = torch.tensor(x_b[id_b], requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, requires_grad=True, dtype=torch.float32).to(device)
    u_f = torch.tensor(u_f[id_f], dtype=torch.float32).to(device)
    u_b_train = torch.tensor(u_b[id_b], dtype=torch.float32).to(device)


    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(x_f_train)
            loss_bc = model.loss_bc(x_b_train, u_b_train)
            loss = loss_pde + loss_bc
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        print(f'epoch {epoch}: loss {loss_value:.6f}')

    print('start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {toc-tic}')

    ## test
    u_test = to_numpy(model(x_test)).reshape(num_test, num_test)
    x_test = to_numpy(x_test)
    u_sol = to_numpy(AC_2D_solu(x_test)).reshape(num_test, num_test)
    print(f'mean test error is :{((u_test-u_sol)**2).mean()}')

    ## plotting
    fig, ax = newfig(2.0, 1.1)
    ax.axis('off') 
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(u_test.T, interpolation='nearest', cmap='rainbow', 
               # extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    fig.colorbar(h)
    ax.plot(x_test[:,1], x_test[:,0], 'kx', label = 'Data (%d points)' % (x_test.shape[0]), markersize = 4, clip_on = False)
    line = np.linspace(x_test.min(), x_test.max(), 2)[:,None]
    savefig('./u_test')

    fig, ax = newfig(2.0, 1.1)
    ax.axis('off') 
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(u_sol.T, interpolation='nearest', cmap='rainbow', 
               # extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    fig.colorbar(h)
    # ax.plot(x[:,1], x[:,0], 'kx', label = 'Data (%d points)' % (x.shape[0]), markersize = 4, clip_on = False)
    line = np.linspace(x_test.min(), x_test.max(), 2)[:,None]
    savefig('./u_sol')

if __name__ == '__main__':
    main()


















