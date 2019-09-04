"""
Author : Minglang Yin
    Multi-fidelity Neural Network (MFNN) with gradient constraint
    High-fidelity data: y_h = (6*x-2)**2*sin(12*x-4)
    Low-fidelity data: y_l = A*(6*x-2)**2*sin(12*x-4)+B*(x-0.5)+C, A = 0.5, B = 10, C = -5

Note:   
    no regularization
    large value of loss

Needs to do: 
    add ploting at the end
    rewrite the initialization framework
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time

from torch.autograd import grad, Variable

torch.manual_seed(13572)
np.random.seed(0)

class Model(nn.Module):
    def __init__(self):#, device):
        super(Model, self).__init__()
        self.net_l = nn.Sequential()
        self.net_l.add_module('layer_1', nn.Linear(1, 20))
        self.net_l.add_module('layer_2', nn.Tanh())
        self.net_l.add_module('layer_3', nn.Linear(20, 20))
        self.net_l.add_module('layer_4', nn.Tanh())
        self.net_l.add_module('layer_5', nn.Linear(20, 1))

        self.net_h_nl = nn.Sequential()
        self.net_h_nl.add_module('layer_1', nn.Linear(2, 10))
        self.net_h_nl.add_module('layer_2', nn.Tanh())
        self.net_h_nl.add_module('layer_3', nn.Linear(10, 10))
        self.net_h_nl.add_module('layer_4', nn.Tanh())
        self.net_h_nl.add_module('layer_5', nn.Linear(10, 1))

        self.net_h_l = nn.Sequential()
        self.net_h_l.add_module('layer_1', nn.Linear(2, 1))

        ## xavier_init
        torch.nn.init.xavier_normal_(self.net_l.layer_1.weight)
        torch.nn.init.xavier_normal_(self.net_l.layer_3.weight)
        torch.nn.init.xavier_normal_(self.net_l.layer_5.weight)
        torch.nn.init.xavier_normal_(self.net_h_nl.layer_1.weight)
        torch.nn.init.xavier_normal_(self.net_h_nl.layer_3.weight)
        torch.nn.init.xavier_normal_(self.net_h_nl.layer_5.weight)
        torch.nn.init.xavier_normal_(self.net_h_l.layer_1.weight)

    def forward(self, x_l, x_h):
        y_l = self.net_l(x_l)

        y_l_h = self.net_l(x_h)
        y_h_nl = self.net_h_nl(torch.cat((y_l_h, x_h), dim=1))
        y_h_l = self.net_h_l(torch.cat((y_l_h, x_h), dim=1))
        y_h = y_h_nl + y_h_l + y_l_h
        return y_l, y_h

    
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def data_loader(device):
    Nl = 15
    Nh = 4

    data = scipy.io.loadmat('./mfdata.mat')
    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    xl = data['xl'].flatten()[:, None]
    yl = data['yl'].flatten()[:, None]
    yl_x = data['yl_x'].flatten()[:, None]
    xh = data['xh'].flatten()[:, None]
    yh = data['yh'].flatten()[:, None]

    x_test = torch.tensor(x, dtype=torch.float32).to(device)
    y_test = torch.tensor(y, dtype=torch.float32).to(device)

    #training data for low fidelity
    id_l = np.random.choice(xl.shape[0], Nl, replace=False)
    x_train_l = torch.tensor(xl[id_l], requires_grad=True, dtype = torch.float32).to(device)
    y_train_l = torch.tensor(yl[id_l], dtype = torch.float32).to(device)
    y_train_l_grad = torch.tensor(yl_x[id_l], dtype=torch.float32).to(device)

    #training data for high fidelity
    id_h = np.random.choice(x.shape[0], Nh, replace=False)
    x_train_h = torch.tensor(xh[id_h], requires_grad=True, dtype= torch.float32).to(device)
    y_train_h = torch.tensor(yh[id_h], dtype=torch.float32).to(device)
    
    return x_test, y_test, x_train_l, y_train_l, y_train_l_grad, x_train_h, y_train_h 

def main():
    ## parameters
    lr = 0.001
    device = torch.device(f"cpu")
    epochs = 50000

    ## load data
    x_test, y_test, x_train_l, y_train_l, y_train_l_grad, x_train_h, y_train_h = data_loader(device)

    ## initialize model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            y_pred_l, y_pred_h = model(x_train_l, x_train_h)

            tmp = torch.ones((15, 1), dtype=torch.float32)
            y_pred_l_grad = grad(y_pred_l, x_train_l, tmp, retain_graph=True)

            loss = ((y_pred_l - y_train_l)**2).mean() +\
                ((y_pred_h - y_train_h)**2).mean() +\
                ((y_pred_l_grad[0] - y_train_l_grad)**2).mean()

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

    # testing and save
    y_pred_l_test, y_pred_h = model(x_test, x_test)
    x_test = to_numpy(x_test)
    y_pred_h = to_numpy(y_pred_h)
    y_test = to_numpy(y_test)
    y_pred_l_test = to_numpy(y_pred_l_test)
    y_test = np.concatenate((x_test, y_pred_h, y_test, y_pred_l_test), axis=1)

    y_pred_l, y_pred_h = model(x_train_l, x_train_h)
    ## save low-fidelity gradient
    tmp = torch.ones((15, 1), dtype=torch.float32)
    y_pred_l_grad = grad(y_pred_l, x_train_l, tmp, retain_graph=True)
    y_train_l_grad = to_numpy(y_train_l_grad)
    y_pred_l_grad = to_numpy(y_pred_l_grad[0])
    x_train_l = to_numpy(x_train_l)
    y_l_grad = np.concatenate((x_train_l, y_pred_l_grad, y_train_l_grad), axis=1)
    ## save low-fidelity data
    y_pred_l = to_numpy(y_pred_l)
    x_train_l = to_numpy(x_train_l)
    y_l = np.concatenate((x_train_l, y_pred_l, y_train_l), axis=1)
    ## save high-fidelity data
    y_pred_h = to_numpy(y_pred_h)
    x_train_h = to_numpy(x_train_h)
    y_h = np.concatenate((x_train_h, y_pred_h, y_train_h), axis=1)
    

    np.savetxt('y_test.dat', y_test)
    np.savetxt('y_l.dat', y_l)
    np.savetxt('y_h.dat', y_h)
    np.savetxt('y_l_grad.dat', y_l_grad)


if __name__ == '__main__':
    main()