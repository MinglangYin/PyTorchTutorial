"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn equation (2D).
    
    d(phi)/dt + div*(phi*u) = 1/Pe*(-F'(phi) + eps^2*lap(phi)) + beta(t)

    where:
        u = 1;
        beta(t) = 0;

    Input: [t, x, y]
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
import scipy.io

from plotting import newfig, savefig

torch.manual_seed(123456)
np.random.seed(123456)

eps = 0.02
Pe = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(3, 30))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for num in range(2,10):
            self.net.add_module('linear_layer_%d' %(num), nn.Linear(30, 30))
            self.net.add_module('tanh_layer_%d' %(num), nn.Tanh())
        self.net.add_module('linear_layer_50', nn.Linear(30, 1))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        u = self.net(x)
        u_g = gradients(u, x)[0]
        u_t, u_x, u_y = u_g[:, 0], u_g[:, 1], u_g[:, 2]
        u_xx = gradients(u_x, x)[0][:, 1]
        u_yy = gradients(u_y, x)[0][:, 2]
        F_g = u**3 - u
        loss = u_t + (u_x + u_y) - 1/Pe*(-F_g + eps**2*(u_xx + u_yy))
        return (loss**2).mean()

    def loss_bc(self, x_b, u_b):
        return ((self.net(x_b)-u_b)**2).mean()

    def loss_ic(self, x_i, u_i):
        u_i_pred = self.net(x_i)
        return ((u_i_pred-u_i)**2).mean()

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

def AC_2D_init(x):
    return np.tanh( (0.1-np.sqrt((x[:,1]-0.5)**2 + (x[:,2] - 0.5)**2)) / (np.sqrt(2)*eps) )

def main():
    ## parameters
    device = torch.device(f"cpu")
    epochs =5000
    lr = 0.001

    num_x = 100
    num_y = 100
    num_t = 10
    num_b_train = 10    # boundary sampling points
    num_f_train = 100   # inner sampling points
    num_i_train = 100   # initial sampling points 

    x = np.linspace(0, 1, num=num_x)
    y = np.linspace(0, 1, num=num_y)
    t = np.linspace(0, 1, num=num_t)
    x_grid, y_grid = np.meshgrid(x, y)
    # x_test = np.concatenate((t_grid.flatten()[:,None], x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    x_2d = np.concatenate((x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    xt_init = np.concatenate((np.zeros((num_x*num_y, 1)), x_2d), axis=1)
    u_init = AC_2D_init(xt_init)[:,None]

    x_2d_ext = np.tile(x_2d, [num_t,1])
    t_ext = np.tile(t[:,None], [num_x*num_y, 1])
    xt_2d_ext = np.concatenate((t_ext, x_2d_ext), axis=1)
    
    ## find a smart way to take boundary point
    x_up = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_dw = np.vstack((x_grid[0,:], y_grid[0,:])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[:, -1], y_grid[:, -1])).T
    x_bound = np.vstack((x_up, x_dw, x_l, x_r))

    x_bound_ext = np.tile(x_bound, [num_t, 1])
    t_bound_ext = np.tile(t[:,None], [num_x*4, 1])
    xt_bound_ext = np.concatenate((t_bound_ext, x_bound_ext), axis=1)
    u_bound_ext = -1*np.ones((num_x*4*num_t))[:,None]

    ## sampling
    id_f = np.random.choice(num_x*num_y*num_t, num_f_train)
    id_b = np.random.choice(num_x*4, num_b_train) ## Dirichlet
    id_i = np.random.choice(num_x*num_y, num_i_train)

    x_i = xt_init[id_i, :]
    u_i = u_init[id_i, :]
    x_f = xt_2d_ext[id_f, :]
    x_b = xt_bound_ext[id_b, :]
    u_b = u_bound_ext[id_b, :]

    ## set data as tensor and send to device
    x_f_train = torch.tensor(x_f, requires_grad=True, dtype=torch.float32).to(device)
    x_b_train = torch.tensor(x_b, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(xt_2d_ext, requires_grad=True, dtype=torch.float32).to(device)
    x_i_train = torch.tensor(x_i, dtype=torch.float32).to(device)
    u_i_train = torch.tensor(u_i, dtype=torch.float32).to(device)
    u_b_train = torch.tensor(u_b, dtype=torch.float32).to(device)


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
            loss_ic = model.loss_ic(x_i_train, u_i_train)
            loss = loss_pde + loss_bc + loss_ic
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
    u_test = np.zeros((num_t, num_x, num_y))
    for i in range(0, 6):
        xt = np.concatenate((t[i]*np.ones((num_x*num_y, 1)), x_2d), axis=1)[:,None]
        xt_tensor = torch.tensor(xt, dtype=torch.float32).to(device)
        u_test[i,:,:] = to_numpy(model(xt_tensor)).reshape(num_x, num_y)

        x_test = to_numpy(x_test)

        fig, ax = newfig(2.0, 1.1)
        ax.axis('off') 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(u_test[i,:,:].T, interpolation='nearest', cmap='rainbow', origin='lower', aspect='auto')
        fig.colorbar(h)
        ax.plot(x_test[:,1], x_test[:,2], 'kx', label = 'Data (%d points)' % (x_test.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(x_test.min(), x_test.max(), 2)[:,None]
        savefig('./u_test_'+str(i))

if __name__ == '__main__':
    main()


















