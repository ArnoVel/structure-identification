from functions.generators.generators import *
from functions.miscellanea import _write_nested, _plotter, GridDisplay
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functions.miscellanea import _write_nested, _plotter, GridDisplay, _basic_univar_distplot
from functions.generators.generators import DatasetSampler

from itertools import product
from scipy.interpolate import UnivariateSpline

X = torch.linspace(-5,5,1000).numpy()

def scale_xy(x,func):
    y = func(x) ; y = (y-y.mean(0))/y.std(0)
    x = (x - x.mean(0))/x.std(0)
    return x,y

def round_(val):
    rounder = (lambda x,precision: int(10**precision * x)/(10**precision))
    if any((isinstance(val,list),
            isinstance(val,tuple),
            isinstance(val,np.ndarray),
            isinstance(val,torch.Tensor))):
        return [rounder(v,3) for v in val]
    else:
        return rounder(val,3)

def viz_mechanisms(num=10):
    for _ in range(num):
        ms = MechanismSampler(X) ; mech = ms.RbfGP(bounds=(2,10))
        plt.plot(*scale_xy(X,mech))
    plt.title('Randomized RBF GP Quantile Sums')
    plt.legend()
    plt.show()
    plt.pause(1)

    for _ in range(num):
        ms = MechanismSampler(X) ; mech = ms.SigmoidAM()
        plt.plot(*scale_xy(X,mech))
    plt.title('Sigmoid AM')
    plt.show()
    plt.pause(1)

    for _ in range(num):
        ms = MechanismSampler(X) ; mech = ms.CubicSpline()
        plt.plot(*scale_xy(X,mech))
    plt.title('Cubic Spline')
    plt.show()
    plt.pause(1)

    for _ in range(num):
        ms = MechanismSampler(X) ; mech = ms.tanhSum()
        plt.plot(*scale_xy(X,mech))
    plt.title('Shift/Scale/Amplitude Tanh Sum')
    plt.show()

def viz_cause(num=10):
    i = 0
    def callback(ax,X,i):
        hist_vals, _ = np.histogram(X,bins='auto', density=True)
        sns.distplot(X, ax=ax, color=f'C{i}')
        low_x, up_x, low_y, up_y = X.min()-X.std(), X.max()+X.std(), 0, hist_vals.max()*1.07
        plt.axis([low_x,up_x,low_y,up_y])
        plt.xticks([], []); plt.yticks([], [])
        plt.tight_layout()

    display = GridDisplay(num_items=10, nrows=-1, ncols=5)
    for i in range(num):
        n = 1000 ; s = CauseSampler(sample_size=n)
        X = s.uniform()
        display.add_plot(callback=(lambda ax: callback(ax,X,i)))

    display.fig.suptitle('Uniform', fontsize=20)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    display = GridDisplay(num_items=10, nrows=-1, ncols=5)
    for i in range(num):
        n = 1000 ; s = CauseSampler(sample_size=n)
        X = s.uniform_mixture()
        display.add_plot(callback=(lambda ax: callback(ax,X,i)))

    display.fig.suptitle('Uniform Mixture', fontsize=20)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    display = GridDisplay(num_items=10, nrows=-1, ncols=5)
    for i in range(num):
        n = 1000 ; s = CauseSampler(sample_size=n)
        X = s.gaussian_mixture()
        display.add_plot(callback=(lambda ax: callback(ax,X,i)))

    display.fig.suptitle('Gaussian Mixture', fontsize=20)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    display = GridDisplay(num_items=10, nrows=-1, ncols=5)
    for i in range(num):
        n = 1000 ; s = CauseSampler(sample_size=n)
        X = s.subgaussian_mixture()
        display.add_plot(callback=(lambda ax: callback(ax,X,i)))

    display.fig.suptitle('Sub Gaussian Mixture', fontsize=20)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    display = GridDisplay(num_items=10, nrows=-1, ncols=5)
    for i in range(num):
        n = 1000 ; s = CauseSampler(sample_size=n)
        X = s.supergaussian_mixture()
        display.add_plot(callback=(lambda ax: callback(ax,X,i)))

    display.fig.suptitle('Super Gaussian Mixture', fontsize=20)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    display = GridDisplay(num_items=10, nrows=-1, ncols=5)
    for i in range(num):
        n = 1000 ; s = CauseSampler(sample_size=n)
        X = s.subsupgaussian_mixture()
        display.add_plot(callback=(lambda ax: callback(ax,X,i)))

    display.fig.suptitle('Sub & Super Gaussian Mixture', fontsize=20)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def viz_pair(save=True):
    SEED = 1020
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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
        if save:
            _write_nested(f'./tests/data/fcm_examples/pairs/anm_{anm}_c_{c}_bn_{bn}_m_{m}',
                          callback= lambda fp: plt.savefig(fp,dpi=70))
            #plt.savefig(f'./data/fcm_examples/pairs/anm_{anm}_c_{c}_bn_{bn}_m_{m}', dpi=40)
        else:
            plt.show()


def viz_confouded(save=True):
    SEED = 1020
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
    base_noises = ['normal', 'student', 'triangular', 'uniform',
                   'beta', 'semicircular']
    mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']
    anms = [ False,True]

    for anm,c,bn_x,bn_y,m_x,m_y in product(anms,causes,
                                            base_noises, base_noises,
                                            mechanisms,mechanisms):
        print(f'anm? {anm}, cause: {c}, base_noise: {bn_x,bn_y}, mechanism: {m_x,m_y}')
        DtSpl = ConfoundedDatasetSampler(N=5, n=1000, anm=anm,
                                        base_noise=[bn_x,bn_y],
                                        confounder_type=c,
                                        mechanism_type=[m_x,m_y],
                                        with_labels=False)

        display = GridDisplay(num_items=5, nrows=-1, ncols=5)
        for pair in DtSpl:
            def callback(ax, pair):
                ax.scatter(pair[0],pair[1], s=10, facecolor='none', edgecolor='k')
                idx = np.argsort(DtSpl.pSampler.x_sample)
                ax.scatter(DtSpl.pSampler.x_sample[idx], DtSpl.pSampler.y_sample[idx], facecolor='r', s=14, alpha=0.7)
            display.add_plot(callback=(lambda ax: callback(ax,pair)))
        display.fig.suptitle(f'Confounded: anm? {anm}, cause: {c}, base_noise: {bn_x,bn_y}, mechanism: {m_x,m_y}', fontsize=20)
        display.fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        if save:
            _write_nested(f'./tests/data/fcm_examples/pairs/cdf_anm_{anm}_c_{c}_bn_{bn_x}+{bn_y}_m_{m_x}+{m_y}',
                          callback= lambda fp: plt.savefig(fp,dpi=70))
            #plt.savefig(f'./data/fcm_examples/pairs/anm_{anm}_c_{c}_bn_{bn}_m_{m}', dpi=40)
        else:
            plt.show()

if __name__ == '__main__':
    #viz_cause()
    #viz_mechanisms()
    #viz_pair(save=True)
    viz_confouded(save=False)
