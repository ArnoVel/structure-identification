import torch
from torch import Tensor

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

def tensorize(scalar, device=DEVICE):
    return Tensor([scalar]).to(DEVICE)

def DotProduct(X,Y):
    '''X,Y contains vector observations X_i, Y_j as rows'''
    return X @ Y.t()

def PairwiseSqDist(X, tol=1e-06):
    ''' Computes norm(X_i - X_j)'''
    XX = DotProduct(X,X)
    # compute diagonal more efficiently
    X2 = (X * X).sum(dim=1).unsqueeze(0)
    #careful, transpose AFTER expanding
    D = -2*XX + X2.expand_as(XX) + X2.expand_as(XX).t()
    return D.abs()+tol

class RBF(torch.nn.Module):
    def __init__(self, precompute=False, bandwidth=1.0, **kwargs):
        super(RBF,self).__init__()
        self.bandwidth = tensorize(bandwidth)
        # whether to compute the pairwise dists
        self.precompute = precompute

    def forward(self,X):
        if self.precompute:
            G = X
        else:
            G = PairwiseSqDist(X)

        return (-self.bandwidth * G).exp()

class Matern(torch.nn.Module):
    def __init__(self,precompute=False, bandwidth=1.0, nu=1.5, **kwargs):
        super(Matern,self).__init__()
        self.bandwidth = tensorize(bandwidth)
        self.nu = nu
        self.precompute = precompute

    def forward(self,X):
        if self.precompute:
            G = X.sqrt()
        else:
            G = PairwiseSqDist(X).sqrt()
        if self.nu == 0.5:
            K = (- self.bandwidth * G).exp()
        elif self.nu == 1.5:
            K = self.bandwidth * G * tensorize(3.0).sqrt()
            K = (tensorize(1.0)+K) * (-K).exp()
        elif self.nu == 2.5:
            K = self.bandwidth * G * tensorize(5.0).sqrt()
            K = (tensorize(1.0) + K + K.pow(2) / tensorize(3.0)) * (-K).exp()
        else:
            raise NotImplementedError("Matern kernel for nu greater than\
                                        2.5 impacractical")

class RQ(torch.nn.Module):
    def __init__(self, precompute=False, bandwidth=1.0, alpha=1.0, **kwargs):
        super(RQ,self).__init__()
        self.bandwidth = tensorize(bandwidth)
        self.alpha = tensorize(alpha)
        self.precompute = precompute

    def forward(self,X):
        if self.precompute:
            G = X
        else:
            G = PairwiseSqDist(X)

        K = (tensorize(1.0) + G/(self.alpha*self.bandwidth)).pow(-self.alpha)

        return K


class SumKernels(torch.nn.Module):
    def __init__(self, kernel_list, device=DEVICE):
        ''' supposes a list of base kernels '''
        super(SumKernels,self).__init__()
        self.kernels = kernel_list
        for k in self.kernels:
            k.precompute = True
        self.device = device

    def forward(self,X):
        G = PairwiseSqDist(X)
        SK = torch.zeros(G.shape).to(self.device)

        for k in self.kernels:
            SK += k(G)
        return SK

class SumIdentical(torch.nn.Module):
    ''' implements the sum of the same type of kernel, with different hyperparameters'''
    def __init__(self, kernel, params):
        super(SumIdentical,self).__init__()
        kernels = [kernel(**p, precompute=True) for p in params]
        self.SumK = SumKernels(kernel_list = kernels)

    def forward(self, X):
        return self.SumK(X)

    # no methods to implement, the forward is simply
    # a SumRBF forward
