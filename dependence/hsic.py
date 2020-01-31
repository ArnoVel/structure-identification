import torch
from scipy.stats import gamma
from functions.kernels import RBF, SumIdentical, SumKernels
from functions.operations import (block_matrix, cross_average,
                                  unblock_matrix, numpy)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
defaults = [{'bandwidth':10**(i)} for i in range(-2,2)]

def kernel_list_(kernels,params):
    return [ker(**p, precompute=True) for (ker,p) in zip(kernels,params)]

class HSIC(torch.nn.Module):
    def __init__(self,n,
                    kernels_X=None, params_X=None,
                    kernels_Y=None, params_Y=None,
                    unbiased=False, device=DEVICE):
        super(HSIC, self).__init__()

        params_X = params_X if params_X is not None else defaults
        params_Y = params_Y if params_Y is not None else defaults
        kernels_X = kernels_X if kernels_X is not None else [RBF for p in params_X]
        kernels_Y = kernels_Y if kernels_Y is not None else [RBF for p in params_Y]

        # check cases where only params are set, or kernels
        if len(params_X)!=len(kernels_X):
            kernels_X = [kernels_X[0] for p in params_X]
        if len(params_Y)!=len(kernels_Y):
            kernels_Y = [kernels_Y[0] for p in params_Y]

        self.KerF_X = SumKernels(kernel_list=kernel_list_(kernels=kernels_X,
                                                          params=params_X))
        self.KerF_Y = SumKernels(kernel_list=kernel_list_(kernels=kernels_Y,
                                                          params=params_Y))

        self.unbiased = unbiased
        self.n = n

    def forward(self,X,Y):
        if X.shape[0]!=self.n or Y.shape[0]!=self.n:
            raise ValueError("Number of samples should be equal",X.shape,Y.shape)

        self.K_X, self.K_Y = self.KerF_X(X), self.KerF_Y(Y)
        # following https://arxiv.org/pdf/1406.3852.pdf,
        # fill diagonal with zeros
        if self.unbiased:
            self.K_X = self.K_X.fill_diagonal_(0)
            self.K_Y = self.K_Y.fill_diagonal_(0)
            statistic = ((self.K_X @ self.K_Y).trace() +
                          self.K_X.sum() * self.K_Y.sum() /(self.n-1)/(self.n-2) -
                          (self.K_X @ self.K_Y).sum()/(self.n-2)
                          ) /self.n/(self.n-3)
        else:
            # using https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf
            H = torch.eye(self.n,self.n) - torch.ones(self.n,self.n)/self.n
            H = H.to(DEVICE)
            statistic = (self.K_X @ H @ self.K_Y @ H).trace() / self.n/self.n

        self.statistic = statistic
        return self.statistic

    def GammaProb(self,X,Y, alpha=0.05):
        ## test only defined for n_x = n_y,
        # therefore drop the sample that's too large
        if X.shape[0]!=self.n or Y.shape[0]!=self.n:
            raise ValueError("Number of samples should be equal",X.shape,Y.shape)
        # useful constants
        H = torch.eye(self.n,self.n) - torch.ones(self.n,self.n)/self.n
        H = H.to(DEVICE)
        c_x, c_y = self.KerF_X(torch.zeros(1,1).to(DEVICE)),\
                   self.KerF_Y(torch.zeros(1,1).to(DEVICE))

        self.K_X, self.K_Y = self.KerF_X(X), self.KerF_Y(Y)
        K_X_c, K_Y_c = H @ self.K_X @ H, H @ self.K_Y @ H
        # we approximate the distribution of m*HSIC_b
        statistic = (K_X_c.t() * K_Y_c).sum() / self.n
        # correct bias by eliminating diagonal terms
        varHSIC = (K_X_c * K_Y_c / 6).pow(2).fill_diagonal_(0).sum() /self.n/(self.n-1)
        varHSIC = 72*(self.n-4)*(self.n-5)/(self.n)/(self.n-1)/(self.n-2)/(self.n-3) * varHSIC

        self.K_X = self.K_X.fill_diagonal_(0)
        self.K_Y = self.K_Y.fill_diagonal_(0)
        mean_x, mean_y = self.K_X.sum()/self.n/(self.n-1), self.K_Y.sum()/self.n/(self.n-1)

        meanHSIC = (c_x*c_y + mean_x*mean_y - c_y*mean_x - c_x*mean_y) / self.n
        self.mean, self.var = meanHSIC, varHSIC
        self.alpha, self.beta = numpy(meanHSIC**2/varHSIC), numpy(self.n*varHSIC/meanHSIC)
        self.test_cdf = (lambda x: 1.0 - gamma.cdf(x, a=self.alpha, loc=0, scale=self.beta))
        self.test_thresh = gamma.ppf(1-alpha, a=self.alpha, loc=0, scale=self.beta)
        self.gamma_test_stat = statistic # not the same as stat, it is n*HSIC biased
        return self.gamma_test_stat, self.test_thresh


import torch
from scipy.stats import gamma
from functions.kernels import RBF, SumIdentical, SumKernels
from functions.operations import (block_matrix, cross_average,
                                  unblock_matrix, numpy)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
defaults = [{'bandwidth':10**(i)} for i in range(-2,2)]

def kernel_list_(kernels,params):
    return [ker(**p, precompute=True) for (ker,p) in zip(kernels,params)]

class CopulaHSIC(torch.nn.Module):
    def __init__(self,n,
                    kernels_X=None, params_X=None,
                    kernels_Y=None, params_Y=None,
                    unbiased=False, device=DEVICE):
        super(HSIC, self).__init__()

        params_X = params_X if params_X is not None else defaults
        params_Y = params_Y if params_Y is not None else defaults
        kernels_X = kernels_X if kernels_X is not None else [RBF for p in params_X]
        kernels_Y = kernels_Y if kernels_Y is not None else [RBF for p in params_Y]

        # check cases where only params are set, or kernels
        if len(params_X)!=len(kernels_X):
            kernels_X = [kernels_X[0] for p in params_X]
        if len(params_Y)!=len(kernels_Y):
            kernels_Y = [kernels_Y[0] for p in params_Y]

        self.KerF_X = SumKernels(kernel_list=kernel_list_(kernels=kernels_X,
                                                          params=params_X))
        self.KerF_Y = SumKernels(kernel_list=kernel_list_(kernels=kernels_Y,
                                                          params=params_Y))

        self.unbiased = unbiased
        self.n = n

    def forward(self,X,Y):
        if X.shape[0]!=self.n or Y.shape[0]!=self.n:
            raise ValueError("Number of samples should be equal",X.shape,Y.shape)

        self.K_X, self.K_Y = self.KerF_X(X), self.KerF_Y(Y)
        # following https://arxiv.org/pdf/1406.3852.pdf,
        # fill diagonal with zeros
        if self.unbiased:
            self.K_X = self.K_X.fill_diagonal_(0)
            self.K_Y = self.K_Y.fill_diagonal_(0)
            statistic = ((self.K_X @ self.K_Y).trace() +
                          self.K_X.sum() * self.K_Y.sum() /(self.n-1)/(self.n-2) -
                          (self.K_X @ self.K_Y).sum()/(self.n-2)
                          ) /self.n/(self.n-3)
        else:
            # using https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf
            H = torch.eye(self.n,self.n) - torch.ones(self.n,self.n)/self.n
            H = H.to(DEVICE)
            statistic = (self.K_X @ H @ self.K_Y @ H).trace() / self.n/self.n

        self.statistic = statistic
        return self.statistic

    def GammaProb(self,X,Y, alpha=0.05):
        ## test only defined for n_x = n_y,
        # therefore drop the sample that's too large
        if X.shape[0]!=self.n or Y.shape[0]!=self.n:
            raise ValueError("Number of samples should be equal",X.shape,Y.shape)
        # useful constants
        H = torch.eye(self.n,self.n) - torch.ones(self.n,self.n)/self.n
        H = H.to(DEVICE)
        c_x, c_y = self.KerF_X(torch.zeros(1,1).to(DEVICE)),\
                   self.KerF_Y(torch.zeros(1,1).to(DEVICE))

        self.K_X, self.K_Y = self.KerF_X(X), self.KerF_Y(Y)
        K_X_c, K_Y_c = H @ self.K_X @ H, H @ self.K_Y @ H
        # we approximate the distribution of m*HSIC_b
        statistic = (K_X_c.t() * K_Y_c).sum() / self.n
        # correct bias by eliminating diagonal terms
        varHSIC = (K_X_c * K_Y_c / 6).pow(2).fill_diagonal_(0).sum() /self.n/(self.n-1)
        varHSIC = 72*(self.n-4)*(self.n-5)/(self.n)/(self.n-1)/(self.n-2)/(self.n-3) * varHSIC

        self.K_X = self.K_X.fill_diagonal_(0)
        self.K_Y = self.K_Y.fill_diagonal_(0)
        mean_x, mean_y = self.K_X.sum()/self.n/(self.n-1), self.K_Y.sum()/self.n/(self.n-1)

        meanHSIC = (c_x*c_y + mean_x*mean_y - c_y*mean_x - c_x*mean_y) / self.n
        self.mean, self.var = meanHSIC, varHSIC
        self.alpha, self.beta = numpy(meanHSIC**2/varHSIC), numpy(self.n*varHSIC/meanHSIC)
        self.test_cdf = (lambda x: 1.0 - gamma.cdf(x, a=self.alpha, loc=0, scale=self.beta))
        self.test_thresh = gamma.ppf(1-alpha, a=self.alpha, loc=0, scale=self.beta)
        self.gamma_test_stat = statistic # not the same as stat, it is n*HSIC biased
        return self.gamma_test_stat, self.test_thresh
