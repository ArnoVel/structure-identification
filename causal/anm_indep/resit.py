import torch
import numpy as np
import matplotlib.pyplot as plt

from functions.regressors.gp.models import GaussianProcess
from functions.regressors.gp.kernels import MaternKernel, RBFKernel, WhiteNoiseKernel, Kernel
from dependence.hsic import HSIC
from functions.operations import numpify

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

hsic_keys = ["stat", "gamma_stat", "gamma_thresh", "gamma_pval"]

class ResitGP:
    ''' REgression and Subsequent Independent Test;
        also adds the option to jointly optimize independence
        and goodness of fit. Uses Gaussian Processes as regressors
    '''
    def __init__(self, hsic_kernels=(None,None), hsic_params=(None,None),
                       gp_kernel=None, gp_eps=1e-02, unbiased=False, device=None):
        ''' uses potentially different kernels for HSIC & GPs.
            The way one specifies kernels for each is slightly different.

            arguments:
                - hsic_kernel: should be 2-tuple containing (kernels_X,kernels_Y)
                  as in dependence.hsic.HSIC;
                - hsic_params: 2-tuple containing hyperparameters for the kernels

                the two default to a sum of gaussian kernels with bandwidths as 10^(i)

                - gp_kernel: the kernel to parametrize the GP regressor.
                  needs to be chosen from functions.regressors.gp.kernels
                  default is RBF kernel.

                - eps (float): lower bound for GP hyperparameters, controlled by clamping

                - reg (float): regularization for the GP covariance matrix
        '''
        self._device = device if device is not None else DEVICE
        self._gp_eps = gp_eps

        if gp_kernel is None:
            self._gp_kernel = RBFKernel(device=self._device) + WhiteNoiseKernel(device=self._device)

        else:
            assert isinstance(gp_kernel, Kernel)
            self._gp_kernel = gp_kernel

        if hsic_kernels is None:
            hsic_kernels = (None,None)
        else:
            # only checks if instances not Module, as no abstract class from functions.kernels defined yet
            assert all(k is None or isinstance(k(),torch.nn.Module) for klist in hsic_kernels for k in klist)
            assert len(hsic_kernels) == 2
        if hsic_params is None:
            hsic_params = (None,None)
        else:
            assert len(hsic_params) == 2

        self._hsic_kernels = hsic_kernels
        self._hsic_params = hsic_params

        self._data_is_set = False

    def set_data(self,X,Y, gp_reg=1e-02, normalize=True, hsic_unbiased=False):
        ''' by default gives the result (hsic-test & GP posterior)
            for a random hyperparameter init
        '''
        assert len(X) == len(Y)
        self._X, self._Y = X.type(dtype), Y.type(dtype)
        self._gp_reg = gp_reg
        # by default basic gp nll fitting
        self._gp = GaussianProcess(self._gp_kernel, device=self._device, eps=self._gp_eps)
        self._gp.set_data(self._X, self._Y, reg=self._gp_reg)

        self._hsic_test = HSIC(n=len(X), unbiased=hsic_unbiased,
                               kernels_X=self._hsic_kernels[0],
                               kernels_Y=self._hsic_kernels[1],
                               params_X=self._hsic_params[0],
                               params_Y=self._hsic_params[0])

        self._hsic_results = {  'gp_init':{},
                                'gp_nll_optim':{},
                                'gp_hsic_optim':{}
                                }

        if normalize:
            self._X = (self._X - self._X.mean(0)) / self._X.std(0)
            self._Y = (self._Y - self._Y.mean(0)) / self._Y.std(0)

        self._hsic_results['gp_init'] = {k:v for k,v in zip(hsic_keys, self.forward_hsic())}

        self._data_is_set = True

    def forward_hsic(self, X=None,Y=None, normalize=True):
        X = self._X if X is None else X
        Y = self._Y if Y is None else X

        if normalize:
            X = (X - X.mean(0)) / X.std(0)
            Y = (Y - Y.mean(0)) / Y.std(0)
            F = self._gp(X)
            F = (F - F.mean(0)) / F.std(0)
            _residuals = Y - F
            _residuals = (_residuals - _residuals.mean(0)) / _residuals.std(0)


        self._hsic_stat = numpify(self._hsic_test(X,_residuals)).ravel()
        self._hsic_gam_stat, self._hsic_gam_thresh = self._hsic_test.GammaProb(X, _residuals, alpha=0.05)
        self._hsic_gam_pval = self._hsic_test.pval

        return self._hsic_stat, self._hsic_gam_stat, self._hsic_gam_thresh, self._hsic_gam_pval

    def train_gp(self, losstype='nll', max_iter=300,
                       reg_factor=2.0, nll_factor=0.1,
                       tol=1e-03):
        ''' args are those of functions.regressors.gp 's GaussianProcess.fit method
        '''
        if losstype != self._gp.loss_type:
            self._gp = GaussianProcess(self._gp_kernel, losstype=losstype, device=self._device, eps=self._gp_eps)
            self._gp.set_data(self._X, self._Y, reg=self._gp_reg)
        print(f'\n in train_gp, with losstype={losstype} \n')
        stop_iter = self._gp.fit(max_iter=max_iter, reg_factor=reg_factor,
                                 nll_factor=nll_factor, tol=tol)

    def run_resit_optim(self, X=None, Y=None,  max_iter=400,
                              reg_factor=2.0, nll_factor=0.1,
                              tol=1e-03, hsic_unbiased=False):
        ''' runs all 3 tests:
                - HSIC with random (init) hyperparameters,
                - HSIC with MLE optimized hyperparameters,
                - HSIC with HSIC optimized hyperparameters
        '''
        if not self._data_is_set:
            if X is None or Y is None:
                raise ValueError("Need to set data if X,Y not given", X, Y)
            else:
                self.set_data(X,Y, hsic_unbiased=hsic_unbiased)
                # setting already runs the init params
        train_args = {'max_iter':max_iter, 'reg_factor':reg_factor,
                      'nll_factor':nll_factor, 'tol':tol}

        stop_iter = self.train_gp(losstype='nll', **train_args)
        self._hsic_results['gp_nll_optim'] = {k:v for k,v in zip(hsic_keys, self.forward_hsic())}

        stop_iter = self.train_gp(losstype='hsic', **train_args)
        self._hsic_results['gp_hsic_optim'] = {k:v for k,v in zip(hsic_keys, self.forward_hsic())}

        return self._hsic_results

    def _plot_gp(self, ax=None):
        if ax is None:
            ax = plt
            no_axis = True
        else:
            no_axis = False

        mu, std = self._gp(self._X, return_std=True)
        X = self._X ; Y = self._Y

        X = (X - X.mean(0)) / X.std(0)
        Y = (Y - Y.mean(0)) / Y.std(0)

        mu, std, X, Y = [numpify(v).ravel() for v in (mu, std, X, Y)]
        idx = np.argsort(X)

        ax.scatter(X[idx], Y[idx], label="Sampled points", s=10,
                                    alpha=0.4, color='k', facecolor='none', marker='o')
        ax.plot(X[idx], mu[idx], "r", label="Estimate")
        ax.fill_between(
            X[idx], (mu[idx] - 3 * std[idx]), (mu[idx] + 3 * std[idx]),
            color="pink", label="Three standard deviations", alpha=0.5)

        title = (
                  "HSIC for " + r"$Y-f(X) \perp\!\!\!\!\!\perp X$" +
                 f"\n has value $t={np.round_(self._hsic_stat,2)}$," +
                 f"  p-value $p={np.round_(self._hsic_gam_pval,2)}$"
                 )

        if no_axis:
            plt.title(title)
        else:
            ax.set_title(title)
