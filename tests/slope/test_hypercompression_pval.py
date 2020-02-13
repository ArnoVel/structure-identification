import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as ppr
from scipy.signal import argrelextrema, argrelmax

from causal.slope.slope import SlopeFunction, _function_to_index, _index_to_function
from causal.slope.utilities import (_get_dtype, _nan_to_zero, _parameter_score,
                                    _bin_int_as_array, _gaussian_score_emp_sse,
                                    _set_resolution)
from functions.miscellanea import _write_nested, _plotter, GridDisplay, _basic_univar_distplot
from functions.generators.generators import DatasetSampler

from itertools import product
from scipy.interpolate import UnivariateSpline


SEED = 1020
torch.manual_seed(SEED)
np.random.seed(SEED)

N = 10000


causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
base_noises = ['normal', 'student', 'triangular', 'uniform',
               'beta', 'semicircular']
mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']
anms = [False, True]

for anm,c,bn,m in product(anms,causes,base_noises,mechanisms):
    print(f'anm? {anm}, cause: {c}, base_noise: {bn}, mechanism: {m}')
    DtSpl = DatasetSampler(N=5, n=1000, anm=anm,
                           base_noise=bn,
                           cause_type=c,
                           mechanism_type=m,
                           with_labels=False)
    display = GridDisplay(num_items=5, nrows=-1, ncols=5)

    for pair in DtSpl:
        def callback(ax, pair):
            ax.scatter(pair[0],pair[1], s=10, facecolor='none', edgecolor='k')
            idx = np.argsort(pair[0])
            x,y = pair[0][idx], pair[1][idx] ; spl = UnivariateSpline(x, y)
            x_display = np.linspace(x.min(), x.max(), 1000)
            ax.plot(x_display, spl(x_display), 'r--')
        display.add_plot(callback=(lambda ax: callback(ax,pair)))
    display.fig.suptitle(f'anm? {anm}, cause: {c}, base_noise: {bn}, mechanism: {m}', fontsize=20)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()
