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


# few inits

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
seed(SEED)

causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
base_noises = ['normal', 'student', 'triangular', 'uniform',
               'beta']
mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']
anms = [False, True]

def iterate_basic_anms():
    '''45 basic anm combinations'''
    c_basic = ['gmm', 'subgmm','supgmm','uniform','mixtunif']
    bn_basic = ['normal', 'student', 'uniform']
    m_basic = ['spline','sigmoidam','tanhsum']
    for c,bn,m in product(c_basic,bn_basic,m_basic):
        yield  c,bn,m

def iterate_example_anms():
    ''' 8 anm examples '''
    c_ex = ['gmm', 'mixtunif']
    bn_ex = ['normal', 'uniform']
    m_ex = ['spline','sigmoidam']
    for c,bn,m in product(c_ex,bn_ex,m_ex):
        yield  c,bn,m

def iterate_example_htr():
    ''' 4 htr examples '''
    examples = [    ['gmm','normal','spline'],
                    ['gmm','normal','sigmoidam'],
                    ['mixtunif','uniform','spline'],
                    ['mixtunif','normal','sigmoidam']
                ]
    for (c,bn,m) in examples:
        yield c,bn,m

def generate_foreach_sample_size(cause, base_noise, mechanism, anm,
                                 dataset_per_comb, sample_size_step=100,
                                 max_sample_size=1000):

    max_sample_size = max(1000, max_sample_size)
    for n in range(100, max_sample_size+sample_size_step, sample_size_step):

        DtSpl = DatasetSampler(N=dataset_per_comb, n=n, anm=anm,
                               base_noise=base_noise, cause_type=cause,
                               mechanism_type=mechanism, with_labels=False)
        for pair in DtSpl:
            yield { 'cause': cause,
                    'base_noise':base_noise,
                    'mechanism':mechanism,
                    'anm':anm,
                    'sample_size':n,
                    'sample':[pair],
                    }


def sample_size_benchmark(dataset_per_comb=100 , sample_size_step=100, max_sample_size=1000):
    ''' for each anm, sample data at different sample sizes '''

    rows = []
    for c,bn,m in iterate_example_anms():
        print(10*'=-=')
        print('ANM of type:',c,bn,m)

        for row in generate_foreach_sample_size(c, bn, m, anm=True,
                                                dataset_per_comb=dataset_per_comb,
                                                sample_size_step=sample_size_step,
                                                max_sample_size=max_sample_size):
            rows.append(row)

    for c,bn,m in iterate_example_htr():
        print(10*'=-=')
        print('HTR of type:',c,bn,m)

        for row in generate_foreach_sample_size(c, bn, m, anm=False,
                                                dataset_per_comb=dataset_per_comb,
                                                sample_size_step=sample_size_step,
                                                max_sample_size=max_sample_size):
            rows.append(row)

    return pd.DataFrame(rows)

def distribution_benchmark(dataset_per_comb=100, sample_sizes=[400,1000]):

    rows = []
    for c,bn,m in iterate_basic_anms():
        print(10*'=-=')
        print(c,bn,m)
        for anm in [True,False]:
            for n in sample_sizes:
                DtSpl = DatasetSampler(N=dataset_per_comb, n=n, anm=anm,
                                       base_noise=bn, cause_type=c,
                                       mechanism_type=m, with_labels=False)
                for pair in DtSpl:

                    row = { 'cause': c,
                            'base_noise':bn,
                            'mechanism':m,
                            'anm':anm,
                            'sample_size':n,
                            'sample':[pair],
                            }
                    rows.append(row)

    return pd.DataFrame(rows)

if __name__ == '__main__':

    # df = sample_size_benchmark(dataset_per_comb=100 , sample_size_step=100, max_sample_size=1000)
    df = distribution_benchmark(dataset_per_comb=100)
    df.to_pickle('./tests/data/geom_ot/fake_data/synth_distribution_benchmark.pkl')
