import torch
import numpy as np
import pandas as pd
from time import time
from random import choice, sample, seed
from itertools  import product
from torch.nn.functional import softmax, log_softmax
import seaborn as sns
import matplotlib.pyplot as plt


from functions.generators.generators import *
from fitting.gmm_fit import GaussianMixture
from functions.miscellanea import _write_nested, _plotter, GridDisplay, mem_report
from causal.generative.geometric import CausalGenGeomNet,GenerativeGeomNet
from causal.slope.utilities import _log, _parameter_score, _log_n


#  WARNING: numpy arrays are stored inside of pandas cells using a list wrapper,
#  WARNING: as follows: [some_array]. One needs to use some_cell[0] to obtain the array

# this file is meant to load nll scores & parameters to compute codelengths
# as well as use the codelengths to decide on the causal direction

data_dir = './tests/data/geom_ot/fake_data/'
scores_dir = './tests/data/geom_ot/data_lengths/sl_gmm/'

scores_filepath = scores_dir + "sl_gmm_datalen_sample_size_benchmark" + '.pkl'
model_len_filepath = scores_dir + "sl_gmm_model_len_sample_size_benchmark" + '.pkl'
# sample_filepath = data_dir + "synth_sample_size_benchmark" + ".pkl"
# data_filepath = data_dir + "fake_data_sample_size_benchmark" + ".pkl"
# model_filepath = data_dir + "fake_data_sample_size_models" + '.pkl'

def _model_pscore(state_dict):
    ''' to adapt depending on whether we cast state_dict to numpy arrays or not'''
    # param_flat = torch.cat( [p.detach().flatten() for p in state_dict.values() ] )
    param_flat = np.concatenate( [p.ravel() for p in state_dict.values() ] )
    # return _parameter_score(param_flat).item()
    ps = _parameter_score(param_flat)

    return ps.item() if ps!=0 else ps


def preprocess_df_params(scores_df):
    ''' removes nll cols, merges noise&mech int a new col, and numpizes param dicts '''
    df = scores_df.copy(deep=True)
    df = scores_df[[ col for col in scores_df.columns if 'nll' not in col ]].copy(deep=True)

    df['noise-mech'] = df['base_noise'] + '-' + df['mechanism']

    df['causal_params'] = df['causal_params'].apply(func=(lambda dct: {k:p.detach().cpu().numpy() for k,p in dct.items()}))
    df['anticausal_params'] = df['anticausal_params'].apply(func=(lambda dct: {k:p.detach().cpu().numpy() for k,p in dct.items()}))

    return df

def preprocess_df_no_params(scores_df):

    df = scores_df[[ col for col in scores_df.columns if 'params' not in col ]].copy(deep=True)

    # cast numerical columns to float, were previously size 0 numpy.ndarrays
    nll_cols = [ col for col in df.columns if 'nll' in col ]
    nll_float_cols = df[nll_cols].apply(lambda x: x.astype(float), axis=1)

    df.loc[:,nll_cols] = nll_float_cols

    # merge test nll cols and add a boolean for causal/not + anm/htr to facilitate seaborn stuff
    c_df = df[['cause','base_noise', 'mechanism', 'anm', 'sample_size', 'causal_test_nll']].copy(deep=True)
    c_df['anm & dir'] = c_df['anm'].apply((lambda b: 'anm' if b else 'htr')) + " & causal"
    c_df['direction'] = ['causal']*c_df.shape[0]
    c_df.rename(columns={'causal_test_nll':'nll'}, inplace=True)

    ac_df = df[['cause','base_noise', 'mechanism', 'anm', 'sample_size', 'anticausal_test_nll']].copy(deep=True)
    ac_df['anm & dir'] = ac_df['anm'].apply(func=(lambda b: 'anm' if b else 'htr')) + " & anticausal"
    ac_df['direction'] = ['anticausal']*ac_df.shape[0]
    ac_df.rename(columns={'anticausal_test_nll':'nll'}, inplace=True)

    plot_df = pd.concat([c_df, ac_df])
    plot_df['noise-mech'] = plot_df['base_noise'] + '-' + plot_df['mechanism']

    return plot_df

# plotting utilities

def lineplot_direction_nll(plot_df):
    ''' just a lineplot comparing nll as fun of sample size depending on causal/not '''

    sns.set(style="whitegrid", palette="muted", font_scale=1)
    sns.lineplot(x="sample_size", y="nll", data=plot_df, marker='o', hue='anm & dir', palette=sns.color_palette(["#0000CD", "#1E90FF", "#FF0000", "#FF8C00"]))

    plt.title("Synthetic GeomNet+GMM NLL vs sample size")
    plt.ylabel("NLL Score")
    plt.show()

def lineplot_dir_nll_foreach_dist(plot_df):
    ''' multiple lineplots of nll as fun of sample size depending on distribution type '''
    sns.set(style="whitegrid", palette="muted", font_scale=0.7)
    g = sns.FacetGrid(plot_df[plot_df.anm == True], row='cause', col='noise-mech', hue="direction", height=3, aspect=1.3)
    g.map(sns.lineplot, "sample_size", "nll")
    g.add_legend();
    # plt.tight_layout()
    plt.subplots_adjust(top=0.91, hspace=0.2, wspace=0.3)
    plt.suptitle("Synthetic GeomNet+GMM: NLL versus sample size (ANM)", fontsize=11)
    plt.ylabel("NLL Score")

    # plt.legend(fontsize=11)
    plt.show()

    g = sns.FacetGrid(plot_df[plot_df.anm == False], row='cause', col='noise-mech', hue="direction", height=3, aspect=1.3)
    g.map(sns.lineplot, "sample_size", "nll")
    g.add_legend();
    # plt.tight_layout()
    plt.subplots_adjust(top=0.91, hspace=0.2, wspace=0.3)
    plt.suptitle("Synthetic GeomNet+GMM: NLL versus sample size (HTR)", fontsize=11)
    plt.ylabel("NLL Score")

    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0, fontsize=11)
    plt.show()

def compute_paramscore(df):
    c_pscores, ac_pscores = [], []
    df['causal_pscore'] = df['causal_params'].apply(func=(lambda p: _model_pscore(p)))
    df['anticausal_pscore'] = df['anticausal_params'].apply(func=(lambda p: _model_pscore(p)))

    df = df.drop("causal_params",1)
    df = df.drop("anticausal_params",1)
    return df

def compute_store_pscore(df):

    save_filepath = scores_dir + "sl_gmm_model_len_sample_size_benchmark" + '.pkl'

    scores_df = pd.read_pickle(scores_filepath)
    df = preprocess_df_params(scores_df)

    df = compute_paramscore(df)

    df.to_pickle(save_filepath)

def normalize_pscores(model_df, method="min"):
    ''' takes the basic pscore dataframe, and normalizes the pscores '''
    if method == "min":
        min_score = min(model_df.loc[model_df['causal_pscore'] > 0,'causal_pscore'].min(),
                        model_df.loc[model_df['anticausal_pscore'] > 0,'anticausal_pscore'].min())
        model_df.loc[model_df['causal_pscore'] > 0,'causal_pscore'] -= min_score
        model_df.loc[model_df['anticausal_pscore'] > 0,'anticausal_pscore'] -= min_score

        return model_df, min_score
    elif method == "mean":
        mean_score = 0.5*(model_df['causal_pscore'].mean() + model_df['anticausal_pscore'].mean())
        model_df['causal_pscore'] /= mean_score
        model_df['anticausal_pscore'] /= mean_score

        return model_df, mean_score
    else:
        raise NotImplementedError("Undefined method name", method)

def preprocess_paramscores(model_df):
    ''' changes the basic pscore dataframe (before or after normalization),
        by adding a direction column and renaming & merging pscore cols'''

    print(model_df.columns)
    c_df = model_df[[   'cause', 'base_noise', 'mechanism',
                        'anm', 'sample_size', 'p', 'num_iters',
                        'num_hiddens', 'noise-mech', 'causal_pscore'
                        ]].copy(deep=True)

    c_df['direction'] = ['causal']*c_df.shape[0]
    c_df.rename(columns={'causal_pscore':'pscore'}, inplace=True)

    ac_df = model_df[[  'cause', 'base_noise', 'mechanism',
                        'anm', 'sample_size', 'p', 'num_iters',
                        'num_hiddens', 'noise-mech', 'anticausal_pscore'
                        ]].copy(deep=True)

    ac_df['direction'] = ['anticausal']*ac_df.shape[0]
    ac_df.rename(columns={'anticausal_pscore':'pscore'}, inplace=True)

    return pd.concat([c_df,ac_df])

def param_len_plot(model_df, method="min"):

    model_df, ref = normalize_pscores(model_df, method=method)

    sns.set(style="whitegrid", palette="muted", font_scale=1)
    # sns.lineplot(x="sample_size", y="nll", data=model_df, marker='o', hue='anm & dir', palette=sns.color_palette(["#FF0000", "#FF8C00", "#0000CD", "#1E90FF"]))
    sns.lineplot(x="sample_size", y="causal_pscore", data=model_df, marker='o', label='Causal')
    sns.lineplot(x="sample_size", y="anticausal_pscore", data=model_df, marker='o', label='Anticausal')

    plt.title("Synthetic GeomNet+GMM: Model Length (GMM params) vs sample size")
    plt.ylabel(f"Model Length (given {np.round(ref,1)} free bits)") ; plt.legend()
    # plt.ylabel(f"Normalized Model Length (given {np.round(ref,1)} average bits)") ; plt.legend()

    plt.show()

def param_len_foreach_dist_plot(model_df, method="min"):

    df, ref = normalize_pscores(model_df, method=method)
    df = preprocess_paramscores(df)

    sns.set(style="whitegrid", palette="muted", font_scale=0.7)

    for b in [True, False]:

        g = sns.FacetGrid(df[df.anm == b], row='cause', col='noise-mech', hue="direction", height=3, aspect=1.3)
        g.map(sns.lineplot, "sample_size", "pscore")
        g.add_legend();
        # plt.tight_layout()
        plt.subplots_adjust(top=0.91, hspace=0.2, wspace=0.3)

        plt.suptitle(f"Synthetic GeomNet+GMM: Model Length (GMM params) vs sample size ({'ANM' if b else 'HTR'})", fontsize=11)
        plt.ylabel(f"Model Length (given {np.round(ref,1)} free bits)")

        # plt.legend(fontsize=11)
        plt.show()


if __name__ == '__main__':
    scores_df = pd.read_pickle(scores_filepath)
    # scores_df = preprocess_df_params(scores_df)
    # model_df = pd.read_pickle(model_len_filepath)

    # param_len_foreach_dist_plot(model_df)

    # param_len_plot(model_df)
    # compute_store_pscore(scores_df)
    plot_df = preprocess_df_no_params(scores_df)
    lineplot_direction_nll(plot_df)
