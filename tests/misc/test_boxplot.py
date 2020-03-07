import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import  pandas as pd
import torch
from itertools import product
from os import listdir
from functions.miscellanea import _write_nested, _plotter, GridDisplay



def load_array(filepath):
    with open(filepath,'rb') as f:
        r = np.load(f)
    return r

def anm_name_parser(name):
    return list(set(name.split('_')) - set(['c','bn','m']))
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

def _basic_benchmark_boxplot(data, ax=None, legend=True, font_scale=1.0):

    df = _dataset_map(data)

    sns.set(style="whitegrid", palette="muted", font_scale=font_scale)
    if ax is None:
        b = sns.boxplot(x="direction", y="value", hue="test", data=df)
        if legend:
            plt.legend(bbox_to_anchor=(1, 1), fancybox=True, fontsize=12,)
        else:
            b.legend_.remove()
    else:
        b = sns.boxplot(x="direction", y="value", hue="test", data=df, ax=ax)
        if legend:
            ax.legend(bbox_to_anchor=(1, 1), fancybox=True, fontsize=12,)
        else:
            b.legend_.remove()

# careful with relative paths and modules
# data = load_array('./tests/data/geom_ot/data_lengths/synthetic/c_gmm_m_spline_bn_normal')

def _all_synthetic_boxplots():
    dirname = './tests/data/geom_ot/data_lengths/synthetic'
    display_idx = 1 ; num_items = len(list(listdir(dirname)))
    display = GridDisplay(num_items=num_items, nrows=-1, ncols=10)
    # for now can't remove legend from every subplot...
    for f in listdir(dirname):
        data = load_array(dirname+'/'+f)
        def callback(ax,data,f,legend):
            _basic_benchmark_boxplot(data, ax=ax, legend=legend, font_scale=0.5)
            ax.set_title('-'.join(anm_name_parser(f)), fontsize=8)

        if display_idx == num_items:
            display.add_plot(callback=(lambda ax: callback(ax,data,f, legend=True)))
            handles, labels = display.last_ax.get_legend_handles_labels()
            display.fig.legend(handles, labels, loc='lower right', fontsize=14)
            display.last_ax.get_legend().remove()
            print('update legend')
        else:
            display.add_plot(callback=(lambda ax: callback(ax,data,f, legend=False)))
            display_idx +=1

    display.fig.tight_layout(pad=3)
    plt.show()


def _tcep_boxplots(num_hiddens=20, max_iter_factor=8):
    ''' single boxplot for data-length using different 2ST's
    '''
    dirname = './tests/data/geom_ot/data_lengths/tcep'
    data = load_array(dirname+'/'+f'tcep_pairs_nh_{num_hiddens}_itfac_{max_iter_factor}')
    _basic_benchmark_boxplot(data)

def _aggregate_datasets(dirname='./tests/data/geom_ot/data_lengths/tcep',
                        rescale_tests=True):
    ''' given data-lengths datasets for each `num_hiddens` and
        `max_iter_factor` values, aggregates all into one dataset,
        by adding two cols
    '''

    cat_dfs = None
    for max_iter_factor, num_hiddens in product(np.arange(5,15,3), np.arange(5,30,5)):
        print(f'(mif,nh) == ({max_iter_factor},{num_hiddens})')

        data = load_array(dirname+'/'+f'tcep_pairs_nh_{num_hiddens}_itfac_{max_iter_factor}')
        df = _dataset_map(data, rescale_tests=rescale_tests)
        df['max_iter_factor'] = [max_iter_factor]*df.shape[0]
        df['num_hiddens'] = [num_hiddens]*df.shape[0]

        cat_dfs = df if cat_dfs is None else pd.concat([cat_dfs,df])
    return cat_dfs

def _tcep_hyperparams_boxplots(x="num_hiddens", restrict_vals=('max_iter_factor',8), restrict=False):
    ''' plots data-lengths vs #epochs or vs num_hiddens,
        given one specifies the x-axis, one can set restrictions
        on the other free param (optional).
    '''

    sns_muted_rgb = sns.color_palette("muted")
    df = _aggregate_datasets(rescale_tests=False)

    assert x in df.columns

    if restrict:
        assert x != restrict_vals[0] ; assert  restrict_vals[0] in df.columns
        df = df[ df[restrict_vals[0]] == restrict_vals[1] ]
    sns.set(style="whitegrid", palette="muted")
    sns.boxplot(x=x, y="value", hue="test", data=df, showfliers=False)
    for test_name, color in zip(df['test'].unique(),sns_muted_rgb):
        plt.axhline(df[df.test == test_name]['value'].median(), color=color, linestyle='dashed')
    plt.show()

def various_boxplots_tcep_hyps_no_dir():
    ''' example of 4 boxplots on tcep
    '''
    _tcep_hyperparams_boxplots(x="num_hiddens", restrict=False)
    _tcep_hyperparams_boxplots(x="max_iter_factor", restrict=False)

    _tcep_hyperparams_boxplots(x="num_hiddens", restrict=True, restrict_vals=('max_iter_factor', 8))
    _tcep_hyperparams_boxplots(x="max_iter_factor", restrict=True, restrict_vals=('num_hiddens', 20))

def _tcep_hyperparams_lineplots_withdir(x='num_hiddens', test='mmd-gamma', ax=None):
    df = _aggregate_datasets(rescale_tests=False)
    sns.set_style("whitegrid")
    assert x in df.columns ; assert test in df['test'].unique()
    df = df[df['test']==test]
    if ax is None:
        sns.lineplot(x=x, y="value",hue="direction", data=df, marker='o')
        plt.xlim([df[x].min(), df[x].max()])
        plt.xticks(df[x].unique().astype('int'))
        plt.title(f'Test: {test}')
    else:
        sns.lineplot(x=x, y="value",hue="direction", data=df, marker='o', ax=ax)
        ax.set_xlim([df[x].min(), df[x].max()])
        ax.set_xticks(df[x].unique().astype('int'))
        ax.set_title(f'Test: {test}')

def _tcep_all_hyperparams_lineplots_withdir():
    tests = ["mmd-gamma", "c2st-knn", "c2st-nn", "test_loss"]
    x_vals = ["num_hiddens", "max_iter_factor"]
    display = GridDisplay(num_items=len(x_vals)*len(tests), nrows=-1, ncols=2)

    for t,xv in product(tests,x_vals):
        display.add_plot(callback = ( lambda  ax: _tcep_hyperparams_lineplots_withdir(x=xv, test=t, ax=ax)))

    plt.show()
if __name__=='__main__':
    # _all_synthetic_boxplots()
    # _tcep_boxplots() ; plt.legend(bbox_to_anchor=(1,1) )
    # plt.savefig('./tests/data/geom_ot/data_lengths/plots/tcep_datalen_boxplots', bbox_inches='tight')
    # various_boxplots_tcep_hyps_no_dir()
    # _tcep_hyperparams_lineplots_withdir()
    # plt.show()
    _tcep_all_hyperparams_lineplots_withdir() ; plt.show()
