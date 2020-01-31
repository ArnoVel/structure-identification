import torch
import numpy as np
from functions.kernels import SumIdentical, RBF, RQ
from dependence.mmd import MMD
from functions.miscellanea import ruled_print, _pickle, _unpickle
import argparse, os

device = torch.device('cuda')
parser = argparse.ArgumentParser(description='MMD-Gamma on 1D distributions')
parser.add_argument('--save', default=0, type=int)
parser.add_argument('--reps', default=1, type=int)

def numpy(tensor):
    return tensor.cpu().numpy()

def two_sample(n,m, mus=[0,0], sigmas=[1,1.5]):
    spX, spY = torch.distributions.normal.Normal(loc=mus[0],scale=sigmas[0]),\
                torch.distributions.normal.Normal(loc=mus[1],scale=sigmas[1])
    X, Y = spX.sample(sample_shape=(n,2)), \
           spY.sample(sample_shape=(m,2))
    return X,Y

def mmd_test(n,m, kernels, params, reps=1,
            save=False, mus=[0,0], sigmas=[1,1.5]):
    mmd_u = MMD(n_x=n,n_y=m, kernels=kernels, params=params,
                   unbiased=True).to(device)
    mmd_b = MMD(n_x=n,n_y=m, kernels=kernels, params=params,
                   unbiased=False).to(device)
    experiments = []
    for _ in range(reps):
        X,Y = two_sample(n,m, mus=mus, sigmas=sigmas)
        X,Y = X.to(device), Y.to(device)
        curr_experiment = [mmd_b(X,Y), mmd_u(X,Y), *mmd_b.GammaProb(X,Y, alpha=0.05)]
        curr_experiment = [numpy(val) if isinstance(val,torch.Tensor) else val.ravel()[0] for val in curr_experiment]
        curr_experiment = np.array(curr_experiment)
        experiments.append(curr_experiment)

    return experiments

def main(args):
    n = 10 ** 3
    m = int(0.7*n)


    params = [{'bandwidth':10**(i),
                'alpha':(4+i)
                } for i in range(-2,2)]
    kernels = [RQ for p in params]
    #print(kernels,params)
    if args.save:
        save = {'params':[], 'test_values':[]}

    # test between gradually less similar normals
    for eps in torch.linspace(0,10,25):
        print('\n\n')
        ruled_print(f'Testing Normal(0,1) versus Normal(0,{1+eps}) using MMD variants')
        mus=[0,0]
        sigmas=[1,1+eps]
        results = mmd_test(n,m, kernels, params, reps=args.reps, save=args.save, mus=mus, sigmas=sigmas)
        mmd_b, mmd_u, gamma_stat, gamma_test_thresh = results[0]
        print(f'Biased MMD^2 : {mmd_b} | versus Unbiased MMD (possibly negative) : {mmd_u}')
        ruled_print(f'Gamma Approximation to the H0 distribution at alpha = 0.05 : is {gamma_stat} >, {gamma_test_thresh} ?')

        if args.save:
            save['params'].append([[mus, sigmas]])
            save['test_values'].append(results)

    if args.save:
        # apparently program is run from the main folder
        _pickle(save, './tests/data/mmd/mmd_test')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
