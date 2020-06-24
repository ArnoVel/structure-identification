import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import  pandas as pd
import torch
from itertools import product
from os import listdir
from cdt.data import load_dataset
from functions.miscellanea import _write_nested, _plotter, GridDisplay
from functions.tcep_utils import _area_under_acc_curve, _threshold_score_pairs, _accuracy_curve, _critical_curve


def load_array(filepath):
    with open(filepath,'rb') as f:
        r = np.load(f)
    return r

def anm_name_parser(name):
    return list(set(name.split('_')) - set(['c','bn','m']))

def dir_name_map(dir):
    assert dir == "<-" or dir == "->"
    return r"$X\to Y$" if (dir == "->") else r"$Y\to X$"

# arrays are stacked along third dim.
# we want first dim stacks, &
def row_map(data,i,c,t):
    t_map = {"mmd-gamma":0,"c2st-nn":1,"c2st-knn":2, "test_loss":3}
    c_map = {r"$X\to Y$":0, r"$Y\to X$":1}
    cm = c_map[c] ; tm = t_map[t]
    assert cm < 2 ; assert tm < 4
    return [c, t, data[tm,cm,i]]

def _dataset_map(data_array, rescale_tests=True):
    tests = ["mmd-gamma","c2st-nn","c2st-knn", "test_loss"]
    # dir = ["X->Y", "Y->X"]
    dir = [r"$X\to Y$", r"$Y\to X$"]

    data = data_array
    print(f'num of NaN-containing Experiments: {np.isnan(data).sum()}')
    data = data[:,:,~np.isnan(data).any(axis=(0,1))]

    if rescale_tests:
        # rescale each test values all together, regarless of causal direction
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - data[i,:,:].min())/(data[i,:,:].max()-data[i,:,:].min())

    df = [ row_map(data,i,c,t) for i,c,t in product(range(data.shape[2]), dir, tests)]
    df = pd.DataFrame(df, columns=['direction','test','value'])

    return df

def _tcep_data(num_hiddens=20, max_iter_factor=8,
               dirname='./tests/data/geom_ot/data_lengths/tcep'):

    ''' single tcep experiment for data-length using different 2ST's
    '''

    data = load_array(dirname+'/'+f'tcep_pairs_nh_{num_hiddens}_itfac_{max_iter_factor}')
    return data

def _aggregate_datasets(dirname='./tests/data/geom_ot/data_lengths/tcep',
                        rescale_tests=True):
    ''' given data-lengths datasets for each `num_hiddens` and
        `max_iter_factor` values, aggregates all into one dataset,
        by adding two cols
    '''

    cat_dfs = None
    for max_iter_factor, num_hiddens in product(np.arange(5,15,3), np.arange(5,30,5)):
        print(f'(mif,nh) == ({max_iter_factor},{num_hiddens})')

        data = _tcep_data(num_hiddens=num_hiddens,max_iter_factor=max_iter_factor,
                          dirname=dirname)

        df = _dataset_map(data, rescale_tests=rescale_tests)
        df['max_iter_factor'] = [max_iter_factor]*df.shape[0]
        df['num_hiddens'] = [num_hiddens]*df.shape[0]

        cat_dfs = df if cat_dfs is None else pd.concat([cat_dfs,df])
    return cat_dfs

def tcep_acc_curves_datalength(max_iter_factor=8, num_hiddens=20, ax=None):
    if ax is None:
        ax = plt

    data, labels = load_dataset("tuebingen",shuffle=False) ; labels = labels.values
    results = _aggregate_datasets(rescale_tests=True)
    results = results[ (results["max_iter_factor"] == max_iter_factor) & (results["num_hiddens"] == num_hiddens)]

    for test in ["mmd-gamma","c2st-nn","c2st-knn", "test_loss"]:
        t_res = results[results["test"] == test ]
        scores = np.vstack([t_res[t_res["direction"] == dir_name_map("->")]["value"].values,
                            t_res[t_res["direction"] == dir_name_map("<-")]["value"].values]).T

        acc_curve = _accuracy_curve(scores,labels)
        ax.plot(acc_curve[:,0], acc_curve[:,1], label=f'Test={test} (AUAC={np.round_(_area_under_acc_curve(scores,labels),2)})')
    crits_dr, crits_vals = _critical_curve(max_n=len(data))
    ax.fill_between(crits_dr,crits_vals,1-crits_vals, alpha=0.5, color='lightgrey')
    ax.axhline(0.5,color='grey',linestyle="--")
    plt.axis([0,1,0,1])
    # print( results[results["direction"] == dir_name_map("->")])
    # print(results[ results[]])
    # for test in results["test"].unique():

if __name__=='__main__':
    dirpath = './tests/data/geom_ot/data_lengths/plots/acc_curves/'
    for max_iter_factor, num_hiddens in product(np.arange(5,15,3), np.arange(5,30,5)):
        tcep_acc_curves_datalength(max_iter_factor, num_hiddens)
        plt.legend() ; plt.savefig(dirpath+f'acc_curve_mif_{max_iter_factor}_nh_{num_hiddens}', dpi=100)
        plt.cla() ; plt.clf()
