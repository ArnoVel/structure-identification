import numpy as np
import matplotlib.pyplot as plt
import time
from random import choices, choice, seed
from itertools import product
from imageio import imread
from matplotlib import pyplot as plt
import torch
from geomloss import SamplesLoss
from functions.generators.generators import *
from functions.miscellanea import _write_nested, _plotter, GridDisplay
from dependence import c2st
from causal.generative.geometric import CausalGenGeomNet,GenerativeGeomNet
from causal.slope.utilities import _log, _parameter_score


use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def convert_scalar(tensor):
    if isinstance(tensor,torch.Tensor):
        return tensor.cpu().flatten().item()
    else:
        return tensor

def _score_wrapper(net):
    param_flat = torch.cat([p.detach().flatten() for p in net.parameters()])
    return _parameter_score(param_flat)

def enumerate_all_anms():
    causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
    base_noises = ['normal', 'student', 'triangular', 'uniform',
                   'beta', 'semicircular']
    mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']
    for i,(c,m,bn) in enumerate(product(causes, mechanisms, base_noises)):
        yield i,c,m,bn

def dataset_geom_net_datalosses(N,cause,mechanism,base_noise):
    DtSpl = DatasetSampler(N=N, n=1000, anm=True,
                           base_noise=bn,
                           cause_type=c,
                           mechanism_type=m,
                           with_labels=False)
    t_start = time.time()
    dataset_results = None
    for idx,XY in enumerate(DtSpl):
        XY = torch.from_numpy(XY)
        XY = XY.type(dtype); X,Y = XY[:,0].clone(), XY[:,1].clone()
        causal_geom_net = CausalGenGeomNet(loss="sinkhorn", p=1, max_iter_factor=8) # iter more than 100 epochs
        causal_geom_net.set_data(X,Y)
        causal_geom_net.fit_two_directions()

        results = None
        for t in ["mmd-gamma","c2st-nn","c2st-knn"]:
            data_prob = causal_geom_net.data_probability(test_type=t, num_tests=5)
            data_prob = [convert_scalar(p) for p in data_prob]
            data_len = np.array([-_log(p) for p in data_prob])
            results = data_len if results is None else np.vstack([results,data_len])

        # result is a [num_probs,2] array
        test_losses = causal_geom_net.test_loss()
        test_losses = [convert_scalar(tl) for tl in test_losses]
        loss_len = np.array([-_log(l) for l in test_losses])
        results = np.vstack([results,loss_len])
        # print(f'parameter compression: X --> Y {_score_wrapper(causal_geom_net._fcm_net_causal)}')
        # print(f'parameter compression: Y --> X {_score_wrapper(causal_geom_net._fcm_net_anticausal)}')
        # stacks matrices along third dim: access mat_i using [:,:,i]
        dataset_results = results if dataset_results is None else np.dstack([dataset_results,results])
        print(f'------- end test for sample {idx}/{N} (i/N) -------')
    t_stop = time.time()
    print(f'Benchmarking dataset with N={N}: elapsed {t_stop-t_start}s , {(t_stop-t_start)/N} s/dataset')
    return dataset_results

def set_seeds(val):
    torch.manual_seed(val)
    np.random.seed(val)
    seed(val)

set_seeds(102)


max_num_tests = np.inf ; N = 200
for i,c,m,bn in enumerate_all_anms():
    if i > max_num_tests:
        break
    else:
        print(f'pair #{i} is: {c}, {m}, {bn}')
        result = dataset_geom_net_datalosses(N=N,cause=c,mechanism=m,base_noise=bn)

        with open(f"tests/data/geom_ot/data_lengths/c_{c}_m_{m}_bn_{bn}", "wb") as f:
            np.save(f,result)
