from functions.generators.generators import *
from functions.miscellanea import _write_nested, _plotter, GridDisplay
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        ms = MechanismSampler(X) ; mech = ms.MaternGP(bounds=(2,10))
        plt.plot(*scale_xy(X,mech))
    plt.title('Randomized Shift/Scale/Amplitude Matern 5/2 Sum')
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

if __name__ == '__main__':
    viz_cause()
