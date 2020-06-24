import torch
import numpy as np
import pandas as pd
from random import choice, sample, seed
from itertools  import product
from torch.nn.functional import softmax, log_softmax
import seaborn as sns
import matplotlib.pyplot as plt


from functions.generators.generators import *
from fitting.gmm_fit import GaussianMixture
from functions.miscellanea import _write_nested, _plotter, GridDisplay, mem_report
from causal.generative.geometric import CausalGenGeomNet,GenerativeGeomNet

# inits
# path from project root (structure-identification/)
data_dir = './tests/data/geom_ot/fake_data/'
# data_filepath = data_dir + "synth_sample_size_benchmark" + ".pkl"
data_filepath = data_dir + "fake_data_sample_size_benchmark" + ".pkl"
model_filepath = data_dir + "fake_data_sample_size_models" + '.pkl'

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

#  WARNING: numpy arrays are stored inside of pandas cells using a list wrapper,
#  WARNING: as follows: [some_array]. One needs to use some_cell[0] to obtain the array

df = pd.read_pickle(data_filepath)
df_models = pd.read_pickle(model_filepath)


def gmm_fit_row(    row, num_mixtures=30, sparsity=1,
                    num_iters=200, display=False,
                    save_figure=False):

    true_pair = torch.from_numpy(row['causal_sample'][0]).type(dtype).t()

    # causal fit
    c_pair = torch.from_numpy(row['causal_sample'][0]).type(dtype).t()
    c_model = GaussianMixture(num_mixtures,sparsity=sparsity, D=c_pair.shape[1])
    c_model.train(c_pair, num_iters=num_iters)

    # anticausal fit
    ac_pair = torch.from_numpy(row['anticausal_sample'][0]).type(dtype).t()
    ac_model = GaussianMixture(num_mixtures,sparsity=sparsity, D=ac_pair.shape[1])
    ac_model.train(ac_pair, num_iters=num_iters)

    c_fake_nll = c_model.neglog_likelihood(c_pair)
    ac_fake_nll = ac_model.neglog_likelihood(ac_pair)
    c_true_nll = c_model.neglog_likelihood(true_pair)
    ac_true_nll = ac_model.neglog_likelihood(true_pair)

    print(f"causal synthetic likelihood on causal fake data: {c_fake_nll}")
    print(f"anticausal synthetic likelihood on anticausal fake data: {ac_fake_nll}")

    print(f"synthetic likelihoods on real data: causal nll = {c_true_nll} | anticausal nll = {ac_true_nll}")

    if display:
        plt.subplot(1,2,1)
        c_model.plot(true_pair)
        plt.title(f"Causal SL", fontsize=10)
        plt.subplot(1,2,2)
        ac_model.plot(true_pair)
        plt.title(f"Anticausal SL", fontsize=10)
        # plt.suptitle("Synthetic Likelihoods (GeomNet+GMM)", fontsize=15)

        if save_figure:
            dirname = "./tests/data/geom_ot/synth_likelihoods/"
            plt.savefig(dirname+ f"sl_gmm_mixtnum_{num_mixtures}_spars_{sparsity}_c_{row['cause']}_bn_{row['base_noise']}_m_{row['mechanism']}_ss_{row['sample_size']}")
            plt.clf()
        else:
            plt.show()

        return None
    data_cols = ["sample" , "causal_sample", "anticausal_sample"]

    ll_row = {col:row[col] for col in row.to_dict() if col not in data_cols}
    ll_row['causal_train_nll'] = c_fake_nll.detach().cpu().numpy()
    ll_row['causal_test_nll'] = c_true_nll.detach().cpu().numpy()
    ll_row['anticausal_train_nll'] = ac_fake_nll.detach().cpu().numpy()
    ll_row['anticausal_test_nll'] = ac_true_nll.detach().cpu().numpy()
    ll_row['causal_params'] = c_model.state_dict()
    ll_row['anticausal_params'] = ac_model.state_dict()

    return ll_row



def sample_and_compute_ll(df, n=30, display=False, save_figure=False):

    for i,row in df.sample(n=n).iterrows():
        print('-----',i,'-----')
        gmm_fit_row(row, display=display, save_figure=save_figure)

def sample_store_ll(df, n=30):
    ll_rows = []
    for i,row in df.sample(n=n).iterrows():
        print('-----',i,'-----')
        row = gmm_fit_row(row)
        ll_rows.append(row)

    return pd.DataFrame(ll_rows)

def compute_store_ll(df):
    ll_rows = []
    for i,row in df.iterrows():
        print('-----',i,'-----')
        row = gmm_fit_row(row)
        ll_rows.append(row)

    return pd.DataFrame(ll_rows)

if __name__ == '__main__':
    # sample_and_compute_ll(df, n=10, display=True, save_figure=True)
    # df = sample_store_ll(df, n=10)
    ll_dir = './tests/data/geom_ot/data_lengths/sl_gmm/'
    ll_df = compute_store_ll(df)
    ll_df.to_pickle(ll_dir + 'sl_gmm_datalen_sample_size_benchmark' + '.pkl' )
