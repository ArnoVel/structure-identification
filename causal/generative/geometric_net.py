import numpy as np
import matplotlib.pyplot as plt
import time
from random import choices, choice, seed
from imageio import imread
from matplotlib import pyplot as plt
import torch
from geomloss import SamplesLoss
from functions.generators.generators import *
from functions.miscellanea import _write_nested, _plotter, GridDisplay
from dependence import c2st

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class GenGeometricNet:
    def __init__(loss="sinkhorn", p=2, blur=1e-01, lr=5e-02, scaling=0.7,
                 max_iter_factor=5, num_hiddens=20):
        ''' each net will be trained for 1/lr * max_iter_factor epochs,
            scaling refers to the sinkhorn loop iterations inside GeomLoss's SamplesLoss,
            and all other arguments are related to networks/SamplesLoss
        '''

        # default values of scaling are taken to be > 0.5, to spend more time
        # in the multiscale sinkhorn loop, and avoir negative sinkhorn loss values
        # see: https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_epsilon_scaling.html
        self.num_iters = int(max_iter_factor/lr) +1
        self.lr = lr ; self.loss = loss ; self.p = p ; self.blur = blur ; self.scaling = scaling
        self.num_hiddens = num_hiddens

    def set_data(self, X,Y):
        ''' X and Y data, each in R^1 for now. (we simplify the problem to point clouds)'''
        assert len(X) == len(Y)
        self._X, self._Y = X.clone(), Y.clone()
        self._X = self._X if self._X.ndim == 2 else self._X.view(-1,1)
        self._Y = self._Y if self._Y.ndim == 2 else self._Y.view(-1,1)
        self._XY = torch.cat([self._X, self._Y], 1)
        self._X, self._Y, self._XY = self._X.type(dtype), self._Y.type(dtype), self._XY.type(dtype)

        self._fcm_net_causal = torch.nn.Sequential(torch.nn.Linear(2,self.num_hiddens),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(self.num_hiddens,self.num_hiddens),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(self.num_hiddens,1)
                                    )
        self._fcm_net_anticausal = torch.nn.Sequential(torch.nn.Linear(2,self.num_hiddens),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(self.num_hiddens,self.num_hiddens),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(self.num_hiddens,1)
                                    )
        self._fcm_net_causal.register_buffer('noise',torch.Tensor(len(self._X), 1))
        self._fcm_net_anticausal.register_buffer('noise',torch.Tensor(len(self._Y), 1))
        self._fcm_net_causal = self._fcm_net_causal.type(dtype)
        self._fcm_net_anticausal = self._fcm_net_anticausal.type(dtype)
        self.optim_causal = torch.optim.Adam(self._fcm_net_causal.parameters(), lr=self.lr)
        self.optim_anticausal = torch.optim.Adam(self._fcm_net_anticausal.parameters(), lr=self.lr)
        self.L_causal = SamplesLoss(loss=self.loss, p=self.p, blur=self.blur, scaling=self.scaling)
        self.L_anticausal = SamplesLoss(loss=self.loss, p=self.p, blur=self.blur, scaling=self.scaling)

    def fit_two_directions(self):

        for i in range(self.num_iters):
            # set grads to zero, re-fill random noise buffer
            self.optim_causal.zero_grad() ; self.optim_anticausal.zero_grad()
            self._fcm_net_causal.noise.normal_() ; self._fcm_net_anticausal.noise.normal_()
            # compute forward quantities
            self._Y_hat = self._fcm_net_causal( torch.cat( [self._X, self._fcm_net_causal.noise], 1 ) )
            self._X_hat = self._fcm_net_anticausal( torch.cat( [self._Y, self._fcm_net_anticausal.noise], 1 ) )
            self._XY_hat_causal = torch.cat( [self._X, self._Y_hat], 1 )
            self._XY_hat_anticausal = torch.cat( [self._X_hat, self._Y], 1)
            # compute loss & backward passes
            loss_causal = self.L_causal(self._XY, self._XY_hat_causal)
            loss_anticausal = self.L_anticausal(self._XY, self._XY_hat_anticausal)
            


        # NOT FINISHED
