import numpy as np
import matplotlib.pyplot as plt
import time
from random import choices, choice, seed
from itertools import product
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import torch
from geomloss import SamplesLoss
from cdt.data import load_dataset
from functions.generators.generators import *
from functions.miscellanea import _write_nested, _plotter, GridDisplay
from dependence import c2st
from causal.generative.geometric import CausalGenGeomNet,GenerativeGeomNet
from causal.slope.utilities import _log, _parameter_score
from functions.tcep_utils import cut_num_pairs,ensemble_score, _get_wd



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
                   'beta']
    mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']
    for i,(c,m,bn) in enumerate(product(causes, mechanisms, base_noises)):
        yield i,c,m,bn

def inner_loop_datalosses(X,Y, loss="sinkhorn", p=1, max_iter_factor=8, num_hiddens=20):
    ''' train & test networks using different two-sample tests / heuristics
    '''
    causal_geom_net = CausalGenGeomNet(loss=loss, p=p,
                                       max_iter_factor=max_iter_factor,
                                       num_hiddens=num_hiddens) # iter more than 100 epochs
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

    return results

def dataset_geom_net_datalosses(N,cause,mechanism,base_noise, anm=True):
    DtSpl = DatasetSampler(N=N, n=1000, anm=anm,
                           base_noise=bn,
                           cause_type=c,
                           mechanism_type=m,
                           with_labels=False)
    t_start = time.time()
    dataset_results = None
    for idx,XY in enumerate(DtSpl):
        XY = torch.from_numpy(XY)
        XY = XY.type(dtype); X,Y = XY[:,0].clone(), XY[:,1].clone()

        results = inner_loop_datalosses(X,Y, loss="sinkhorn", p=1)
        # print(f'parameter compression: X --> Y {_score_wrapper(causal_geom_net._fcm_net_causal)}')
        # print(f'parameter compression: Y --> X {_score_wrapper(causal_geom_net._fcm_net_anticausal)}')
        # stacks matrices along third dim: access mat_i using [:,:,i]
        dataset_results = results if dataset_results is None else np.dstack([dataset_results,results])
        print(f'------- end test for sample {idx}/{N} (i/N) -------')
    t_stop = time.time()
    print(f'Benchmarking dataset with N={N}: elapsed {t_stop-t_start}s , {(t_stop-t_start)/N} s/dataset')
    return dataset_results

def tcep_geom_net_datalosses(max_sample_size=(10**3), max_iter_factor=8, num_hiddens=20):
    data , labels = load_dataset('tuebingen',shuffle=False)
    cut_num_pairs(data,num_max=max_sample_size)

    dataset_results = None
    for i,row in data.iterrows():
        X,Y = process(row)
        X,Y = torch.from_numpy(X).type(dtype) , torch.from_numpy(Y).type(dtype)

        results = inner_loop_datalosses(X,Y, loss="sinkhorn", p=1)

        # stacks matrices along third dim: access mat_i using [:,:,i]
        dataset_results = results if dataset_results is None else np.dstack([dataset_results,results])
        print(f'------- end test for sample {i}/{len(data)} (i/N) -------')
    return dataset_results


def process(row):
    x,y = (scale(row['A']), scale(row['B']))
    return x,y

def set_seeds(val):
    torch.manual_seed(val)
    np.random.seed(val)
    seed(val)

set_seeds(102)

def _run_anm_tests(last_covered,N, anm=True):
    for i,c,m,bn in enumerate_all_anms():
        print(f'pair #{i} is: {c}, {m}, {bn}')
        if i > max_num_tests:
            break
        elif i>last_covered:
            result = dataset_geom_net_datalosses(N=N,cause=c,mechanism=m,base_noise=bn, anm=anm)
            # print(result)
            with open(f"tests/data/geom_ot/data_lengths/synthetic/c_{c}_m_{m}_bn_{bn}", "wb") as f:
                np.save(f,result)

def _run_tcep_tests(max_iter_factor=8, num_hiddens=20):
    results = tcep_geom_net_datalosses(max_iter_factor=max_iter_factor, num_hiddens=num_hiddens)
    with open(f"tests/data/geom_ot/data_lengths/tcep/tcep_pairs_nh_{num_hiddens}_itfac_{max_iter_factor}", "wb") as f:
        np.save(f,results)


if __name__=='__main__':
    max_num_tests = np.inf ; N = 200 ; last_covered = 43
    # _run_anm_tests(last_covered,N)
    for mif, nh in product(np.arange(5,15,3), np.arange(5,30,5)):
        print(f'Training for max_iter_factor={mif} & num_hiddens={nh}')
        _run_tcep_tests(max_iter_factor=mif, num_hiddens=nh)
