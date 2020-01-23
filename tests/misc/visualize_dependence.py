import torch
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

import matplotlib.pyplot as plt


def two_gaussians(n, covariance=[1,0,0,1]):
    sampler = MVN(loc=torch.zeros(2),
                  covariance_matrix=torch.Tensor([covariance[0:2],covariance[2:4]]))
    X,Y = sampler.sample((n,)).t()
    return X.view(-1,1), Y.view(-1,1)


rho = .25
X,Y = two_gaussians(5000,covariance=[1,rho,rho,1])

X = -X+.5
Y = Y.pow(3)
plt.scatter(X,Y, s=10, marker='o', facecolor=None, edgecolor='k')
plt.show()
