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
import pprint as ppr

# few inits

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
seed(SEED)

pp = ppr.PrettyPrinter(indent=4)

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

data_dir = './tests/data/geom_ot/fake_data/'
data_filepath = data_dir + "synth_sample_size_benchmark" + ".pkl"

#  WARNING: numpy arrays are stored inside of pandas cells using a list wrapper,
#  WARNING: as follows: [some_array]. One needs to use some_cell[0] to obtain the array

df = pd.read_pickle(data_filepath)

# df_test = df.sample(n=10)

def train_generate_store(df, p=1, max_iter_factor=7.5, num_hiddens=20, lr=5e-02):
    ''' picks every row from the dataframe, trains a GeomNet in each direction,
        generates the same amount of samples, and then creates two new rows
    '''
    rows = []
    state_dicts = []
    for i,row in df.iterrows():
        pair = row['sample'][0]
        XY = torch.from_numpy(pair).type(dtype).t()
        X,Y = XY[:,0].clone(), XY[:,1].clone()
        causal_geom_net = CausalGenGeomNet(loss="sinkhorn", p=p, max_iter_factor=max_iter_factor, num_hiddens=num_hiddens)
        causal_geom_net.set_data(X,Y)
        causal_geom_net.fit_two_directions(display=False)

        assert causal_geom_net.trained
        causal_geom_net._fcm_net_causal.eval()
        causal_geom_net._fcm_net_anticausal.eval()

        causal_geom_net.forward()

        XY_hat_causal = causal_geom_net._XY_hat_causal.detach().cpu().numpy()
        XY_hat_anticausal = causal_geom_net._XY_hat_anticausal.detach().cpu().numpy()

        print(f"pair #{i}, shape:{pair.shape}")
        row = row.to_dict()
        row['causal_sample'] = [XY_hat_causal.T]
        row['anticausal_sample'] = [XY_hat_anticausal.T]
        row['p'] = p ; row['num_iters'] = int(max_iter_factor/lr) ; row['num_hiddens'] = num_hiddens

        c_dict = causal_geom_net._fcm_net_causal.state_dict() ; c_dict['direction'] = 'causal'
        ac_dict = causal_geom_net._fcm_net_anticausal.state_dict() ; ac_dict['direction'] = 'anticausal'
        c_dict['index'] = i ; ac_dict['index'] = i
        state_dicts.append(ac_dict) ; state_dicts.append(c_dict)

        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(state_dicts)

# WARNING: Not sure the CausalGenGeomNet class is equipped to sample from each nets...
# WARNING: Might need to add that as a method later...

if __name__ == '__main__':
    data_df, model_df = train_generate_store(df)
    # for i,row in data_df.iterrows():
    #     print(row['sample'][0].shape, row['anticausal_sample'][0].shape, row['causal_sample'][0].shape)

    data_df.to_pickle(data_dir + 'fake_data_sample_size_benchmark' + '.pkl' )
    model_df.to_pickle(data_dir + 'fake_data_sample_size_models' + '.pkl' )
