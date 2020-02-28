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
from dependence import c2st, mmd

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class CausalGenGeomNet:
    def __init__(self,loss="sinkhorn", p=2, blur=1e-01, lr=5e-02, scaling=0.7,
                 max_iter_factor=5, num_hiddens=20):
        ''' each net will be trained for 1/lr * max_iter_factor epochs,
            scaling refers to the sinkhorn loop iterations inside GeomLoss's SamplesLoss,
            and all other arguments are related to networks/SamplesLoss.

            This method allows to train the two networks jointly at the cost of a more
            cluttered code...
        '''

        # default values of scaling are taken to be > 0.5, to spend more time
        # in the multiscale sinkhorn loop, and avoir negative sinkhorn loss values
        # see: https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_epsilon_scaling.html
        self.num_iters = int(max_iter_factor/lr) +1
        self.lr = lr ; self.loss = loss ; self.p = p ; self.blur = blur ; self.scaling = scaling
        self.num_hiddens = num_hiddens
        self.trained = False ; self.data_is_set = False

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

        self.data_is_set = True

    def forward(self):
        assert self.data_is_set
        self._fcm_net_causal.noise.normal_() ; self._fcm_net_anticausal.noise.normal_()
        # compute forward quantities
        self._Y_hat = self._fcm_net_causal( torch.cat( [self._X, self._fcm_net_causal.noise], 1 ) )
        self._X_hat = self._fcm_net_anticausal( torch.cat( [self._Y, self._fcm_net_anticausal.noise], 1 ) )
        self._XY_hat_causal = torch.cat( [self._X, self._Y_hat], 1)
        self._XY_hat_anticausal = torch.cat( [self._X_hat, self._Y], 1)

    def fit_two_directions(self):
        ''' fits the two networks using the same number of epochs '''

        self._fcm_net_causal.train() ; self._fcm_net_anticausal.train()

        for i in range(self.num_iters):
            # set grads to zero, re-fill random noise buffer
            self.optim_causal.zero_grad() ; self.optim_anticausal.zero_grad()

            self.forward()
            # compute loss & backward passes
            loss_causal = self.L_causal(self._XY, self._XY_hat_causal)
            loss_anticausal = self.L_anticausal(self._XY, self._XY_hat_anticausal)

            loss_causal.backward() ; loss_anticausal.backward()
            self.optim_causal.step() ; self.optim_anticausal.step()

            if i and not i%50:
                print(f' optim step #{i}: Loss X->Y is {loss_causal.data} | Loss Y->X is {loss_anticausal.data}')

        self.trained = True

    def data_probability(self, test_type="mmd-gamma", num_tests=1):
        ''' Given a base sample S and a fitted model f(-,N), evaluates the probability
            that S comes from the model using a two sample test.
            In the case of a bivariate causal relationship (X_i,Y_i), evaluates the
            sample probability that Y_i = f(X_i,N).
            Conversely, assuming Y_i = f(X_i,N), this enquires about the probability of the data.

            The mmd-gamma test requires cdf from scipy, therefore not torch differentiable
            (torch gamma cdf not implemented yet)
        '''
        assert self.trained
        self._fcm_net_causal.eval() ; self._fcm_net_anticausal.eval()

        self.forward()

        self._XY = self._XY.detach() # somehow attached?
        self._XY_hat_causal = self._XY_hat_causal.detach()
        self._XY_hat_anticausal = self._XY_hat_anticausal.detach()

        self.causal_stat, self.causal_pval = _get_test(self._XY, self._XY_hat_causal, test_type, return_test=False)
        self.anticausal_stat, self.anticausal_pval = _get_test(self._XY, self._XY_hat_anticausal, test_type, return_test=False)

        if num_tests > 1:
            for _ in range(num_tests-1):
                self.forward()
                # somehow needs to be further detached?
                self._XY_hat_causal = self._XY_hat_causal.detach()
                self._XY_hat_anticausal = self._XY_hat_anticausal.detach()

                # average stats & pvals over num_tests trials
                c_s, c_pv = _get_test(self._XY, self._XY_hat_causal, test_type, return_test=False)
                self.causal_stat += c_s ; self.causal_pval += c_pv
                ac_s, ac_pv = _get_test(self._XY, self._XY_hat_anticausal, test_type, return_test=False)
                self.anticausal_stat += ac_s ; self.anticausal_pval += ac_pv

            self.causal_stat, self.causal_pval,  = self.causal_stat/num_tests, self.causal_pval/num_tests
            self.anticausal_stat, self.anticausal_pval = self.anticausal_stat/num_tests, self.anticausal_pval/num_tests

        return self.causal_pval, self.anticausal_pval

    def test_loss(self, num_tests=200):
        assert self.trained
        self._fcm_net_causal.eval() ; self._fcm_net_anticausal.eval()
        self.forward()
        test_loss_causal = self.L_causal(self._XY, self._XY_hat_causal).data
        test_loss_anticausal = self.L_anticausal(self._XY, self._XY_hat_anticausal).data

        for _ in range(num_tests-1):
            self.forward()
            test_loss_causal += self.L_causal(self._XY, self._XY_hat_causal).data
            test_loss_anticausal += self.L_anticausal(self._XY, self._XY_hat_anticausal).data

        test_loss_causal /= num_tests ; test_loss_causal = test_loss_causal.cpu()
        test_loss_anticausal /= num_tests ; test_loss_anticausal = test_loss_anticausal.cpu()

        return test_loss_causal, test_loss_anticausal

    def data_compression(self, test_type='mmd-gamma'):
        ''' returns the two part codelength of Y|X implicitly assuming the model holds
        '''
        pass
        # NOT FINISHED


# a base class for a single task of distribution regression X-->Y
class GenerativeGeomNet:
    def __init__(self,loss="sinkhorn", p=2, blur=1e-01, lr=5e-02, scaling=0.7,
                 max_iter_factor=5, num_hiddens=20):
        ''' Only one geometric generative model.
            scaling refers to the sinkhorn loop iterations inside GeomLoss's SamplesLoss,
            and all other arguments are related to networks/SamplesLoss.
        '''
        # default values of scaling are taken to be > 0.5, to spend more time
        # in the multiscale sinkhorn loop, and avoir negative sinkhorn loss values
        # see: https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_epsilon_scaling.html
        self.num_iters = int(max_iter_factor/lr) +1 ; self.max_iter_factor = max_iter_factor
        self.lr = lr ; self.loss = loss ; self.p = p ; self.blur = blur ; self.scaling = scaling
        self.num_hiddens = num_hiddens
        self.trained = False ; self.data_is_set = False

        self._net = torch.nn.Sequential(torch.nn.Linear(2,self.num_hiddens),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.num_hiddens,self.num_hiddens),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.num_hiddens,1))

        self._L = SamplesLoss(loss=self.loss, p=self.p, blur=self.blur, scaling=self.scaling)
        self._opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)


    def set_data(self,X,Y):
        ''' X and Y data, each in R^1 for now. (we simplify the problem to point clouds)'''
        assert len(X) == len(Y)
        self._X, self._Y = X.clone(), Y.clone()
        self._X = self._X if self._X.ndim == 2 else self._X.view(-1,1)
        self._Y = self._Y if self._Y.ndim == 2 else self._Y.view(-1,1)
        self._XY = torch.cat([self._X, self._Y], 1)
        self._X, self._Y, self._XY = self._X.type(dtype), self._Y.type(dtype), self._XY.type(dtype)
        self._net.register_buffer('noise',torch.Tensor(len(self._X), 1))
        self._net = self._net.type(dtype)

        self.data_is_set = True

    def forward(self):
        assert self.data_is_set
        self._net.noise.normal_()
        # compute forward quantities
        self._Y_hat = self._net( torch.cat( [self._X, self._net.noise], 1 ) )
        self._XY_hat = torch.cat( [self._X, self._Y_hat], 1)

    def train(self, num_iters=None):
        ''' The number of training iterations defaults to 1/lr * max_iter_factor
            (given at init). This can be changed by passing the num_iters argument()
        '''
        if num_iters is not None:
            self.num_iters = num_iters
        for i in range(self.num_iters):
            # set grads to zero, re-fill random noise buffer
            self._opt.zero_grad() ; self.forward()
            # compute loss & backward passes
            loss = self._L(self._XY, self._XY_hat)
            loss.backward() ; self._opt.step()

            if i and not i%50:
                print(f' optim step #{i}: Loss is {loss.data}')
        self.trained = True

    def reset_net(self):
        self._net = torch.nn.Sequential(torch.nn.Linear(2,self.num_hiddens),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.num_hiddens,self.num_hiddens),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.num_hiddens,1)
                                        )
        self._net.register_buffer('noise',torch.Tensor(len(self._X), 1))
        self._net = self._net.type(dtype)
        self._L = SamplesLoss(loss=self.loss, p=self.p, blur=self.blur, scaling=self.scaling)
        self._opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)


    def data_probability(self, test_type="mmd-gamma", num_tests=1):
        ''' Given a base sample S and a fitted model f(-,N), evaluates the probability
            that S comes from the model using a two sample test.
            In the case of a bivariate causal relationship (X_i,Y_i), evaluates the
            sample probability that Y_i = f(X_i,N).
            Conversely, assuming Y_i = f(X_i,N), this enquires about the probability of the data.

            The mmd-gamma test requires cdf from scipy, therefore not torch differentiable
            (torch gamma cdf not implemented yet)
        '''
        assert self.trained
        self._net.eval()
        self.forward()

        self._XY = self._XY.detach() # somehow attached?
        self._XY_hat = self._XY_hat.detach()

        self.stat, self.pval = _get_test(self._XY, self._XY_hat, test_type, return_test=False)

        if num_tests > 1:
            for _ in range(num_tests-1):
                self.forward()

                self._XY = self._XY.detach() # somehow attached?
                self._XY_hat = self._XY_hat.detach()

                stat, pval = _get_test(self._XY, self._XY_hat, test_type, return_test=False)
                self.stat += stat ; self.pval += pval

            self.stat /= num_tests ; self.pval /= num_tests

        return self.stat, self.pval

    def test_loss(self, num_tests=200):
        assert self.trained
        self._net.eval()
        self.forward()
        test_loss = self._L(self._XY, self._XY_hat).data

        for _ in range(num_tests-1):
            self.forward()
            test_loss += self._L(self._XY, self._XY_hat).data

        test_loss /= num_tests ; test_loss = test_loss.cpu()

        return test_loss

    def data_compression(self):
        raise NotImplementedError("Not Yet!")

    def plot_training_samples(self, num_displays=6):
        assert self.data_is_set
        self.reset_net()
        if num_displays > 2 & num_displays < 10:
            ncols = 3
        elif num_displays > 10:
            ncols = 5

        display_its = [int(t/self.lr) for t in torch.linspace(0,self.max_iter_factor,num_displays)]
        display = GridDisplay(num_items=num_displays, nrows=-1, ncols=ncols)

        colors = (10*self._XY[:,0]).cos() * (10*self._XY[:,1]).cos()
        colors = colors.detach().cpu().numpy()

        for i in range(self.num_iters):
            # set grads to zero, re-fill random noise buffer
            self._opt.zero_grad() ; self.forward()
            # compute loss & backward passes
            loss = self._L(self._XY, self._XY_hat)
            loss.backward() ; self._opt.step()

            if i in display_its : # display
                print(f'loop display iter #{i}')

                def callback(ax, x_i, y_j, colors):
                    plt.set_cmap("hsv")
                    display_samples(ax, self._XY, [(.55,.55,.95)])
                    display_samples(ax, self._XY_hat, colors)
                    ax.set_title("t = {:1.2f}".format(self.lr*i))

                    plt.xticks([], []); plt.yticks([], [])
                    plt.tight_layout()
                display.add_plot(callback=(lambda ax: callback(ax, self._XY, self._XY_hat, colors)))

        if self.loss == "sinkhorn":
            L_info = {"loss":self.loss, "p":self.p, "blur":self.blur,"scaling":self.scaling}
        else:
            L_info = {"loss":self.loss, "blur":self.blur}
        display.fig.suptitle((f"Gradient flows with loss {L_info};"+
                              f"\n until T = {self.lr*i} (steps of {int(1e03/self.lr)/1e03})"),
                              fontsize=10)
        display.fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.show()


def _get_test(P,Q, test_name, return_test=False):
    assert len(P) == len(Q)
    n = len(P)
    if test_name == 'mmd-gamma':
        # default is a sum of RBF kernels with bandwidths as 10**(i) for i in (-2..2)
        test = mmd.MMD(n,n) ; test.GammaProb(P,Q)
        if return_test:
            return test
        else:
            return test.gamma_test_stat, test.pval
    elif test_name =='c2st-nn':
        return c2st.neural_net_c2st(P,Q, return_test=return_test)
    elif test_name=='c2st-knn':
        return c2st.knn_c2st(P,Q, return_test=return_test)
    else:
        raise NotImplementedError("test type:",test_name)

def display_samples(ax, x, color) :
    plt.set_cmap("hsv")
    x_ = x.detach().cpu().numpy()
    ax.scatter( x_[:,0], x_[:,1], 25*500 / len(x_), color, edgecolors='none' )
