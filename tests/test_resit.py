import torch
import numpy as np
import pprint as ppr
from random import seed, choice
import matplotlib.pyplot as plt

from causal.anm_indep.resit import ResitGP
from functions.generators.generators import DatasetSampler
from functions.kernels import SumIdentical, RBF, RQ

pp = ppr.PrettyPrinter(indent=4)

def anm_data(num_samples=1000, SEED=0):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    seed(SEED)
    causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
    base_noises = ['normal', 'student', 'triangular', 'uniform',
                   'beta']
    mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']

    global c,bn,m
    c,bn,m = choice(causes), choice(base_noises), choice(mechanisms)
    print(f'random choice of ANM: {c,bn,m}')
    DtSpl = DatasetSampler(N=1, n=num_samples, anm=True,
                           base_noise=bn,
                           cause_type=c,
                           mechanism_type=m,
                           with_labels=False)
    DtSpl.__iter__() ; pair = torch.from_numpy(next(DtSpl))

    return pair

def anm_dataset(N=10,num_samples=1000,SEED=0, anm_choice=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    seed(SEED)
    causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
    base_noises = ['normal', 'student', 'triangular', 'uniform',
                   'beta']
    mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']

    global c,bn,m
    if anm_choice is None:
        c,bn,m = choice(causes), choice(base_noises), choice(mechanisms)
    else:
        c,bn,m = anm_choice

    print(f'random choice of ANM: {c,bn,m}')
    DtSpl = DatasetSampler(N=N, n=num_samples, anm=True,
                           base_noise=bn,
                           cause_type=c,
                           mechanism_type=m,
                           with_labels=False)

    def dtsp_wrapper():
        for pair in DtSpl:
            yield torch.from_numpy(pair)
    return dtsp_wrapper()

if __name__ == '__main__':
    # resit = RESIT()
    # data = anm_data(SEED=0)
    # print(data.shape)
    # X,Y = data[0].view(-1,1), data[1].view(-1,1)
    # results = resit.run_resit_optim(X,Y)
    # pp.pprint(results)
    # params = [{'bandwidth':10**(i),
    #             'alpha':(4+i)
    #             } for i in range(-2,2)]
    params = [{'bandwidth':10**(i)} for i in range(-6,3)]
    kernels = [RBF for p in params]

    print(kernels)

    for pair in anm_dataset(N=2, SEED=1612):
        resit = ResitGP(hsic_kernels=(kernels,kernels), hsic_params=(params,params))
        X,Y = pair[0].view(-1,1), pair[1].view(-1,1)
        results = resit.run_resit_optim(X,Y, max_iter=200,
                                             reg_factor=2.0,
                                             nll_factor=1e-04,
                                             hsic_unbiased=False)
        pp.pprint(results)
        resit._plot_gp() ; plt.show()
