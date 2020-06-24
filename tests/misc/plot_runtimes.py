import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# set main paths. the paths are from the root of the directory
# as scripts are called using pyt -m tests.some.path for relative imports

dir_runtimes = './tests/data/fitting/runtimes/'
filepath_gmm = dir_runtimes +  'gmm_runtimes' + '.pkl'
filepath_geom = dir_runtimes +  'geomnet_runtimes' + '.pkl'
filepath_geom_large = dir_runtimes + 'geomnet_runtimes' + '_large_with_flip' + '.pkl'

df_gmm = pd.read_pickle(filepath_gmm)
df_geom = pd.read_pickle(filepath_geom)
df_geom_lg = pd.read_pickle(filepath_geom_large)

# stats.gamma.fit finds the triplet (shape, loc, scale) through MLE
# gam_params = stats.gamma.fit(df_geom['loss'])

def estimate_test_loss_dist_gamma(store_data=True):
    gam_params_floc = stats.gamma.fit(df_geom_lg['loss'], floc=0) # fix loc at 0

    x_vals = np.linspace(0,df_geom_lg['loss'].max(), 1000)
    fit_gam_pdf = stats.gamma.pdf(x_vals, *gam_params_floc)

    sns.distplot(df_geom_lg['loss'], label='Loss histogram')
    plt.plot(x_vals,fit_gam_pdf, 'r', label='Gamma MLE')
    plt.legend()
    plt.show()

    df = pd.DataFrame({ 'loss':'sinkhorn',
                        'p':1,
                        'scaling':0.7,
                        'blur':1e-01,
                        'max_iter_factor':7.5,
                        'num_hiddens':20,
                        'gamma_approx_parameters': [np.array(gam_params_floc)]
                        })
    if store_data:
        df.to_pickle('./tests/data/fitting/ot/test_loss_dist/gamma_approx_parameters.pkl')
    else:
        return df

def plot_runtimes(save_figure=False):
    sns.set(style="whitegrid", palette="muted", font_scale=1.0)

    df_gmm['algorithm'] = ['gmm']* df_gmm.shape[0]
    df_geom['algorithm'] = ['geom net']* df_geom.shape[0]

    df = pd.concat([df_gmm, df_geom])
    df['runtime'] = df['runtime']
    bp = sns.boxplot(x="algorithm", y="runtime", data=df)
    if save_figure:
        pass
    else:
        plt.ylabel('runtime (s)')
        plt.xlabel('algorithm type')
        plt.show()


if __name__ == '__main__':
    # plot_runtimes(False)
    estimate_test_loss_dist_gamma(store_data=True)
