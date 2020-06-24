import torch
import numpy as np
import pandas as pd
from random import choice, sample, seed
from itertools  import product
from torch.nn.functional import softmax, log_softmax
from functions.generators.generators import *

from fitting.gmm_fit import GaussianMixture
from functions.miscellanea import _write_nested, _plotter, GridDisplay, mem_report
from causal.generative.geometric import CausalGenGeomNet,GenerativeGeomNet
from matplotlib import pyplot as plt
import seaborn as sns

from time import time
import GPUtil

# Aims to check experimentally the computational complexity of fitting networks and gmms

# gmm_train_runtime & gmm_store_runtimes just train and store runtimes & associated hyperparameters in a DataFrame
# gmm_fit_dataset & gmm_run_tests give average time per dataset & display loss and likelihood contours.

# the same functions exist for geom_net.

# few inits

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
seed(SEED)

num_combinations, dataset_per_comb = 10, 10
num_iters = 200

causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
base_noises = ['normal', 'student', 'triangular', 'uniform',
               'beta']
mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']
anms = [False, True]

# check cuda
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# shorthands

def gmm_fit_dataset(cause,base_noise,mechanism, anm,
                num_combinations, dataset_per_comb,
                num_mixtures=30, sparsity=1,
                num_iters=200, display=False,
                save_figure=False):

    DtSpl = DatasetSampler(N=dataset_per_comb, n=1000, anm=anm,
                           base_noise=base_noise, cause_type=cause,
                           mechanism_type=mechanism, with_labels=False)

    losses = []
    t_start = time()
    for pair in DtSpl:
        pair = torch.from_numpy(pair).type(dtype).t()
        model = GaussianMixture(num_mixtures,sparsity=sparsity, D=pair.shape[1])
        model.train(pair, num_iters=num_iters)
        #print(model.neglog_likelihood(pair))
        losses.append(model.loss)


    t_stop = time()
    print(f"Average time for GMM fit ({num_mixtures} classes, {1000} points, {dataset_per_comb} runs): {(t_stop - t_start)/dataset_per_comb}")

    if display:
        model.plot(pair)
        if save_figure:
            dirname = "./tests/data/fitting/gmm/dim-two/causal_synth/"
            plt.savefig(dirname+ f"gmm_mixtnum_{num_mixtures}_spars_{sparsity}_{cause}_{base_noise}_{mechanism}_{'anm' if anm else 'htr'}")
        plt.show()


    return losses

def gmm_run_tests(num_combinations, dataset_per_comb, display, save_figure):
    for c,bn,m,anm in sample(list(product(causes,base_noises,mechanisms,anms)), num_combinations):
        print(10*'=-=')
        print(c,bn,m,anm)
        losses = gmm_fit_dataset(c,bn,m,anm, num_combinations, dataset_per_comb, display=display, save_figure=save_figure)

def gmm_train_runtime(cause,base_noise,mechanism, anm,
                num_mixtures=30, sparsity=1, num_iters=200):

    DtSpl = DatasetSampler(N=dataset_per_comb, n=1000, anm=anm,
                           base_noise=base_noise, cause_type=cause,
                           mechanism_type=mechanism, with_labels=False)

    rows = []
    for pair in DtSpl:
        pair = torch.from_numpy(pair).type(dtype).t()
        t_start = time()
        model = GaussianMixture(num_mixtures,sparsity=sparsity, D=pair.shape[1])
        model.train(pair, num_iters=num_iters)
        t_stop = time()
        rows.append({   'cause':cause,
                        'base_noise':base_noise,
                        'mechanism':mechanism,
                        'anm':anm,
                        'num_iters':num_iters,
                        'num_mixtures':num_mixtures,
                        'runtime':t_stop-t_start})

    return pd.DataFrame(rows)


def gmm_store_runtimes(num_combinations, dataset_per_comb, return_df=True):
    ''' if return_df, returns instead of store. stores by default as pickle'''
    dataframes = []
    for c,bn,m,anm in sample(list(product(causes,base_noises,mechanisms,anms)), num_combinations):
        print(10*'=-=')
        print(c,bn,m,anm)
        dataframes.append( gmm_train_runtime(c,bn,m, anm) )

    df = pd.concat(dataframes)
    if return_df:
        return df
    else:
        df.to_pickle('./tests/data/fitting/runtimes/gmm_runtimes.pkl')

def geomnet_train_runtime(cause,base_noise,mechanism, anm,
                          p=1, max_iter_factor=7.5, lr=5e-02,
                          num_hiddens=20, flip=False):

    DtSpl = DatasetSampler(N=dataset_per_comb, n=1000, anm=anm,
                           base_noise=base_noise, cause_type=cause,
                           mechanism_type=mechanism, with_labels=False)

    rows = []
    for pair in DtSpl:
        XY = torch.from_numpy(pair).type(dtype).t()

        XY = XY.type(dtype); X,Y = XY[:,0].clone(), XY[:,1].clone()

        t_start = time()
        # init network
        geom_net = GenerativeGeomNet(loss="sinkhorn", p=p,
                                     max_iter_factor=max_iter_factor,
                                     lr=lr, num_hiddens=num_hiddens)
        if flip:
            geom_net.set_data(Y,X)
        else:
            geom_net.set_data(X,Y)

        geom_net.train()

        t_stop = time()

        rows.append({   'cause':cause,
                        'base_noise':base_noise,
                        'mechanism':mechanism,
                        'anm':anm,
                        'flip':flip,
                        'num_iters':int(max_iter_factor/lr),
                        'num_hiddens':num_hiddens,
                        'loss': float(geom_net.test_loss(num_tests=10)),
                        'runtime':t_stop-t_start})

    return pd.DataFrame(rows)

def geomnet_store_runtimes(num_combinations, dataset_per_comb,
                           return_df=True, id_string=None,
                           flip_each=False):
    ''' if return_df, returns instead of store. stores by default as pickle'''
    dataframes = []
    for c,bn,m,anm in sample(list(product(causes,base_noises,mechanisms,anms)), num_combinations):
        print(10*'=-=')
        print(c,bn,m,anm)
        dataframes.append( geomnet_train_runtime(c,bn,m, anm) )
        if flip_each:
            dataframes.append( geomnet_train_runtime(c,bn,m, anm, flip=True) )

    df = pd.concat(dataframes)
    if return_df:
        return df
    else:
        if id_string is not None:
            df.to_pickle(f'./tests/data/fitting/runtimes/geomnet_runtimes_{id_string}.pkl')
        else:
            df.to_pickle('./tests/data/fitting/runtimes/geomnet_runtimes.pkl')
# main loop
if __name__ == '__main__':
    # gmm_run_tests(num_combinations, dataset_per_comb, False, False)
    # df = gmm_store_runtimes(num_combinations, dataset_per_comb, return_df=False)
    # print(df)
    # df = geomnet_train_runtime('gmm','normal','spline',True,p=1)
    # print(df)
    num_combinations, dataset_per_comb = 25, 20
    df = geomnet_store_runtimes(num_combinations, dataset_per_comb, return_df=False, id_string="large_with_flip", flip_each=True)
