import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from functions.regressors.gp.models import GaussianProcess
from functions.regressors.gp.kernels import RBFKernel, WhiteNoiseKernel
from functions.operations import numpify
from dependence.hsic import HSIC
import torch.distributions as tdist

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


def round_(val, nums_after_point=3):
    return int(val * (10** nums_after_point))/ (10**nums_after_point)

def lapgauss(n, sigma=1):
    D1, D2 = tdist.normal.Normal(0,sigma) , tdist.laplace.Laplace(0,sigma)
    V,W = D1.rsample(sample_shape=(n//2,)), D2.rsample(sample_shape=(n//2,))
    N = torch.cat([V,W])
    idx = torch.randperm(N.nelement())
    return N[idx].unsqueeze(1)

def plot_variance(gp, x, f,y , title=None, std_factor=1.0):
    mu, std = gp(x, return_std=True)
    std *= std_factor
    mu = numpify(mu).ravel()
    std = numpify(std).ravel()

    x = numpify(x).ravel()
    y = numpify(y).ravel()
    f = numpify(f).ravel()
    idx = np.argsort(x)

    plt.scatter(x[idx], y[idx], label="Sampled points",
                                alpha=0.4, color='r', facecolor='none', marker='o')
    plt.plot(x[idx], f[idx], "k--", label="Ground truth")
    plt.plot(x[idx], mu[idx], "r", label="Estimate")
    plt.fill_between(
        x[idx], (mu[idx] - 3 * std[idx]), (mu[idx] + 3 * std[idx]),
        color="grey", label="Three standard deviations", alpha=0.6)
    plt.axis([-4, 4, -2, 2])
    plt.title("Gaussian Process Estimate" if title is None else title)
    plt.legend()
    plt.show()

def print_residual_test(gp, X, Y):
    F = gp(X) ; F = (F - F.mean(0))/F.std(0)
    residuals = Y-gp(X) ; residuals = (residuals - residuals.mean(0))/residuals.std(0)

    hsic = HSIC(n=residuals.nelement(),unbiased=False)

    stat, statgam, threshgam, pval = [
                        numpify(hsic(X,residuals)).ravel(),
                       *hsic.GammaProb(X,residuals, alpha=0.05),
                        hsic.test_cdf(numpify(hsic.gamma_test_stat).ravel())
                        ]
    print((f"Test for X indep Y-f(X) ;\n"+
           f"statistic:{stat},\n" +
           f"gamma (test_stat, thresh): ({statgam},{threshgam}),\n"+
           f"pval:{pval}"))
    return stat, statgam, threshgam, pval

n = 500; X = 4*torch.rand(n,1) - 2
f = (lambda x: 1.5*x.pow(3) - 2*((-x/.5).pow(2)+x)*x.cos())

F = f(X); X = (X - X.mean(0))/X.std(0) ;
F = (F-F.mean(0))/F.std(0)
sigma = np.abs(F.max()-F.min())/10
N = lapgauss(500,sigma=sigma); Y = F+N
Y = (Y - Y.mean(0))/Y.std(0);

X,Y = X.to(DEVICE), Y.to(DEVICE)

ker = RBFKernel(device=DEVICE) + WhiteNoiseKernel(device=DEVICE)
gp = GaussianProcess(ker, device=DEVICE, eps=1e-02)
gp.set_data(X,Y, reg=1e-02)

print('What are the HSIC residual stats before optim?')
stat, statgam, threshgam, pval = print_residual_test(gp, X, Y)
label = ("Before Optimization:\nHSIC for " + r"$Y-f(X) \perp\!\!\!\!\!\perp X$" +
         f"  has value $t={round_(statgam)}$,  p-value $p={round_(pval)}$")
plot_variance(gp, X, F, Y, title=label)

# fit gp hyperparameters
print('Now training...') ; stop = gp.fit(max_iter=400) ; print(f'stop:{stop}')

stat, statgam, threshgam, pval = print_residual_test(gp, X, Y)
label = ("After NLL Optimization:\nHSIC for " + r"$Y-f(X) \perp\!\!\!\!\!\perp X$" +
         f"  has value $t={round_(statgam)}$,  p-value $p={round_(pval)}$")
plot_variance(gp, X, F, Y, title=label)

# now try to optimise hsic directly
# ker_hsic = RBFKernel(device=DEVICE) \
#     + RBFKernel(device=DEVICE) \
#     + WhiteNoiseKernel(device=DEVICE)
# either restart training new kern, or warmastart it using old nll kernel
ker_hsic = RBFKernel(device=DEVICE) + WhiteNoiseKernel(device=DEVICE)
gp_hsic = GaussianProcess(ker, device=DEVICE, losstype="hsic", eps=1e-02)
gp_hsic.set_data(X,Y, reg=1e-02)

print('Now training with HSIC...') ; stop = gp_hsic.fit(max_iter=600, reg_factor=2.0, nll_factor=0.1)
print(f'stop:{stop}')

stat, statgam, threshgam, pval = print_residual_test(gp_hsic, X, Y)
label = ("After HISC Optimization:\nHSIC for " + r"$Y-f(X) \perp\!\!\!\!\!\perp X$" +
         f"  has value $t={round_(statgam)}$,  p-value $p={round_(pval)}$")
plot_variance(gp_hsic, X, F, Y, title=label)
