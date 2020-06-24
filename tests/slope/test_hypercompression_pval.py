import torch
import numpy as np
from scipy import stats,optimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import argrelextrema, argrelmax

from causal.slope.slope import SlopeFunction, _function_to_index, _index_to_function
from causal.slope.utilities import (_get_dtype, _nan_to_zero, _parameter_score,
                                    _bin_int_as_array, _gaussian_score_emp_sse,
                                    _set_resolution)
from functions.miscellanea import (_write_nested, _plotter, GridDisplay,
                                   _basic_univar_distplot,_compare_two_distplot)
from functions.generators.generators import DatasetSampler

from itertools import product
from scipy.interpolate import UnivariateSpline


SEED = 1020
torch.manual_seed(SEED)
np.random.seed(SEED)

N = 10000


causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
base_noises = ['normal', 'student', 'triangular', 'uniform',
               'beta']
mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']
anms = [False, True]

def _find_exponent_nonlin_ls(x,y):
    ''' assume noisy data y = a^(-x) + eps,
        find a, use nonlinear least squares, better fit
    '''
    param_opt, _ = optimize.curve_fit(lambda x,a: np.power(a,-x),  x,  y,  p0=(2.0))
    return param_opt

def _find_exponent_lin_ls(x,y):
    ''' for comparison, quite bad fits due to log space'''
    # if one wants to fit y = 1/a^x => log(y) = - log(a^x) = - log(a)*x
    # => log transform y, and fit a linear function log(y) ~ b*x and posit b ~ -log(a) => a = exp(-b)
    y_pos = y.copy() ; y_pos[np.where(y_pos<=0)] = y[np.where(y>0)].min()
    log_y_pos = np.log(y_pos)
    sol, residuals, rank, singular_vals = np.linalg.lstsq(x.reshape(-1,1),log_y_pos.reshape(-1,1), rcond=None)
    return np.exp(-sol)

def _compare_twofits_example(base=4.27):
    x = np.linspace(1,10, 500) ; y = np.power(base, -x) ; n = np.random.randn(500) * 1e-02
    plt.plot(x,y, 'r--') ; plt.scatter(x,(y+n), s=10, facecolor='none', edgecolor='k')
    sol = _find_exponent_lin_ls(x,y+n)
    approx_base = sol ; plt.plot(x,np.power(approx_base,-x).ravel(), 'g-.', label=f'log-space linfit (a_approx={int(1e03*float(approx_base))/1e03})')
    param_opt = _find_exponent_nonlin_ls(x,y+n)
    plt.plot(x,np.power(param_opt,-x), 'b--', label=f'nonlin fit LS (a_approx={int(1e03*float(param_opt))/1e03})')
    plt.title(f'Noisy exponential a^(-x) with a={base}')
    plt.legend()
    plt.show()
# for anm,c,bn,m in product(anms,causes,base_noises,mechanisms):
#     print(f'anm? {anm}, cause: {c}, base_noise: {bn}, mechanism: {m}')
#     DtSpl = DatasetSampler(N=5, n=1000, anm=anm,
#                            base_noise=bn,
#                            cause_type=c,
#                            mechanism_type=m,
#                            with_labels=False)
    #display = GridDisplay(num_items=5, nrows=-1, ncols=5)

## to see which least squares technique yields best fit
# _compare_twofits_example(base=4.27)
# _compare_twofits_example(base=2.0)
# _compare_twofits_example(base=6.13)
def _bootstrap_hypercompression_bounds(
            x,y, num_bootstrap_its=150,
            bootstrap_samples=600, mixed=True, nofc=10):
    assert len(x) > bootstrap_samples
    score_xy = np.array([]) ; score_yx = np.array([])
    for _ in range(num_bootstrap_its):
        idx_boot = np.random.choice(len(x),bootstrap_samples, replace=True)
        x_b, y_b = x[idx_boot], y[idx_boot]
        # X to Y
        resolution = _set_resolution(x_b) ; slope_f = SlopeFunction(num_functions=nofc)
        if mixed:
            rr = slope_f._find_best_mixed_fit(x_b,y_b)
        else:
            rr = slope_f._find_best_fit_index(x_b,y_b)
        score = _gaussian_score_emp_sse(rr['sse'], len(x_b), resolution=resolution) + rr['model_score']
        score_xy = np.concatenate([score_xy, score])

        # Y to X
        resolution = _set_resolution(y_b) ; slope_f = SlopeFunction(num_functions=nofc)
        if mixed:
            rr = slope_f._find_best_mixed_fit(y_b,x_b)
        else:
            rr = slope_f._find_best_fit_index(y_b,x_b)
        score = _gaussian_score_emp_sse(rr['sse'], len(y_b), resolution=resolution) + rr['model_score']
        score_yx = np.concatenate([score_yx, score])

    return score_xy, score_yx

def _plot_hypcomp_bootstrap(c,bn,n,m, alpha):
    DtSpl = DatasetSampler(N=2, n=n, anm=True,
                               base_noise=bn,
                               cause_type=c,
                               mechanism_type=m,
                               with_labels=False)
    DtSpl.__iter__(); pair = next(DtSpl)

    score_xy, score_yx = _bootstrap_hypercompression_bounds(pair[0], pair[1])

    # assuming correlation, take the null compression length as L0 = (L(XY)+L(YX))/2.
    # then, the probability that any other model L1 (which means L(XY) or L(YX))
    # compresses |L0 - L1| = k bits better "should be 2^{-k}"
    L0 = (score_xy + score_yx)/2
    abs_bit_gap = np.abs(L0 - score_xy)
    _basic_univar_distplot(abs_bit_gap, distname='abs bit gap', mode=False, kde=False, nbins='auto')

    plt.axvline(-np.log2(alpha), color='k', linestyle='-.',
               linewidth=2, label=f'Significance gap'+r" ($\alpha=$"+f"{alpha})")
    plt.legend()
    plt.title(f"Bootstrapped Compression Absolute Bit Gap \n on ANM ({c},{bn},{m})")
    plt.show()
    bit_gap_xy = L0 - score_xy ; bit_gap_yx = L0 - score_yx
    _compare_two_distplot(bit_gap_xy, bit_gap_yx, distnames=['X->Y','Y->X'], mode=False, nbins='auto')
    plt.title(f"Bootstrapped Compression Signed Bit Gap on ANM ({c},{bn},{m})")
    plt.legend() ; plt.show()

def _anm_hypercompression_bounds(
            N,n,c,bn,m, mixed=True, nofc=10):
    DtSpl = DatasetSampler(N=N, n=n, anm=True,
                               base_noise=bn,
                               cause_type=c,
                               mechanism_type=m,
                               with_labels=False)
    score_xy = np.array([]) ; score_yx = np.array([])
    for i,pair in enumerate(DtSpl):
        x,y = pair ; print(i) if not (i+1)%50 else None
        # X to Y
        resolution = _set_resolution(x) ; slope_f = SlopeFunction(num_functions=nofc)
        if mixed:
            rr = slope_f._find_best_mixed_fit(x,y)
        else:
            rr = slope_f._find_best_fit_index(x,y)
        score = _gaussian_score_emp_sse(rr['sse'], len(x), resolution=resolution) + rr['model_score']
        score = score if np.ndim(score) else [score]
        score_xy = np.concatenate([score_xy, score])

        # Y to X
        resolution = _set_resolution(y) ; slope_f = SlopeFunction(num_functions=nofc)
        if mixed:
            rr = slope_f._find_best_mixed_fit(y,x)
        else:
            rr = slope_f._find_best_fit_index(y,x)
        score = _gaussian_score_emp_sse(rr['sse'], len(y), resolution=resolution) + rr['model_score']
        score = score if np.ndim(score) else [score]
        score_yx = np.concatenate([score_yx, score])

    return score_xy, score_yx

def _plot_hypercomp_anm_dataset(N,n,c,bn,m, mixed=True, nofc=10):

    score_xy, score_yx = _anm_hypercompression_bounds(N=N,n=n,c=c,bn=bn,m=m,mixed=mixed, nofc=nofc)

    L0 = (score_xy + score_yx)/2
    abs_bit_gap = np.abs(L0 - score_xy)
    _basic_univar_distplot(abs_bit_gap, distname='abs bit gap', mode=False, kde=False, nbins='auto')

    plt.axvline(-np.log2(alpha), color='k', linestyle='-.',
               linewidth=2, label=f'Significance gap'+r" ($\alpha=$"+f"{alpha})")
    plt.legend()
    plt.title(f"Compression Absolute Bit Gap \n on ANM-Dataset ({c},{bn},{m})")
    plt.show()
    bit_gap_xy = L0 - score_xy ; bit_gap_yx = L0 - score_yx
    _compare_two_distplot(bit_gap_xy, bit_gap_yx, distnames=['X->Y','Y->X'], mode=False, nbins='auto')
    plt.title(f"Compression Signed Bit Gap on ANM-Dataset ({c},{bn},{m})")
    plt.legend() ; plt.show()

c, bn, m = 'gmm','normal','spline' ; alpha = 1e-02 ; n=1000
#_plot_hypcomp_bootstrap(c,bn,n,m,alpha)
_plot_hypercomp_anm_dataset(N=200,n=1000,c=c,bn=bn,m=m,mixed=True)
