import torch
import numpy as np
import matplotlib.pyplot as plt
from functions.generators.generators import CauseSampler, MechanismSampler
from functions.kernels import SumIdentical, RBF, RQ
from dependence.hsic import HSIC
from functions.miscellanea import ruled_print, _pickle, _unpickle, mem_report
import argparse, os

device = torch.device('cuda')
parser = argparse.ArgumentParser(description='HSIC-Gamma on 1D distributions')
parser.add_argument('--save', default=0, type=int)
parser.add_argument('--reps', default=1, type=int)
parser.add_argument('--size', default=10**3, type=int)

def numpy(tensor):
    return tensor.cpu().numpy()

def rescale(vals):
    return (vals - vals.mean(0))/vals.std(0)

def anm_u_spl(n,eps=0.0):
    ''' simulates X ~ gmm and Y = Unif(-1,1) + eps*f(X),
    with f a cubic spline
    '''
    cause_sampler = CauseSampler(sample_size=n).gaussian_mixture
    X = cause_sampler(); mapping = MechanismSampler(X).CubicSpline()
    noise = CauseSampler(sample_size=n).uniform()
    Y = noise + eps*mapping(X)
    Y = (Y - Y.mean()) / Y.std()
    X,Y = torch.from_numpy(X), torch.from_numpy(Y)
    return X.view(-1,1), Y.view(-1,1)

def hsic_test(n, reps=1,
            kernels_X=None, params_X=None,
            kernels_Y=None, params_Y=None,
            save=False, eps=0.0):
    ''' samples X,Y from a 2d-gaussian with various levels of correlation '''
    hsic_u = HSIC(n=n,
                  kernels_X=kernels_X, params_X=params_X,
                  kernels_Y=kernels_Y, params_Y=params_Y,
                  unbiased=True)
    hsic_b = HSIC(n=n,
                  kernels_X=kernels_X, params_X=params_X,
                  kernels_Y=kernels_Y, params_Y=params_Y,
                  unbiased=False)
    experiments = []
    for _ in range(reps):
        X,Y = anm_u_spl(n=1000, eps=eps)
        X,Y = X.to(device), Y.to(device)
        curr_experiment = [hsic_b(X,Y), hsic_u(X,Y),
                           *hsic_b.GammaProb(X,Y, alpha=0.05),
                           hsic_b.test_cdf(numpy(hsic_b.gamma_test_stat))]
        curr_experiment = [numpy(val) if isinstance(val,torch.Tensor) else val.ravel()[0] for val in curr_experiment]
        curr_experiment = np.array(curr_experiment)
        experiments.append(curr_experiment)

    del hsic_b.K_X,hsic_b.K_Y,hsic_b, hsic_u.K_X, hsic_u.K_Y, hsic_u, X,Y
    torch.cuda.empty_cache()
    #mem_report()
    return experiments

def main(args):
    n = args.size


    params_X = [{'bandwidth':10**(i),
                'alpha':(4+i)
                } for i in range(-2,2)]
    params_Y = params_X
    kernels_X = [RBF for p in params_X]
    kernels_Y = [RBF for p in params_Y]
    #print(kernels,params)
    if args.save:
        save = {'params':[], 'test_values':[]}

    # test between gradually less similar normals
    for eps in torch.linspace(0,0.99,25):
        print('\n\n')
        ruled_print(f'Testing X~ GMM, Y = eps*f(X)+ Unif with eps={eps} increasing')
        #mem_report()
        results = hsic_test(n, reps=args.reps,
                            kernels_X=kernels_X, params_X=params_X,
                            kernels_Y=kernels_Y, params_Y=params_Y,
                            save=args.save, eps=numpy(eps))

        hsic_b, hsic_u, gamma_stat, gamma_test_thresh, pval = results[0]
        print(f'Biased HSIC : {hsic_b} | versus Unbiased HSIC (possibly negative) : {hsic_u}')
        ruled_print(f'Gamma Approximation to the H0 distribution at alpha = 0.05 : is {gamma_stat} > {gamma_test_thresh} ?')
        print(f'the p-value is approximated as {pval}')
        if args.save:
            save['params'].append(eps)
            save['test_values'].append(results)

    if args.save:
        # apparently program is run from the main folder
        _pickle(save, './tests/data/hsic/hsic_anm_test')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
