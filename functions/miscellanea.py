''' Random functions useful for file management, display, etc.'''
import pickle
from pathlib import Path
import argparse, os, gc, inspect
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, argrelmax
import seaborn as sns
import pprint as ppr
import torch
import imageio

def ruled_print(string, rule_symbol='-'):
    sentences = string.split('\n')
    max_length = max([len(s) for s in sentences])
    print(max_length*rule_symbol)
    print(string)
    print(max_length*rule_symbol)

def _pickle(dict,fname):
    with open(f'{fname}.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def _unpickle(fname):
    with open(f'{fname}.pickle', 'rb') as handle:
        return pickle.load(handle)

def _write_nested(filepath,callback):
    ''' supposes a callback which only takes
        the filepath as argument to write/save a file;
        creates the nested folders  if needed
    '''
    atoms = filepath.split('/')
    path, file = '/'.join(atoms[:-1]), atoms[-1]
    Path(path).mkdir(parents=True, exist_ok=True)
    callback(filepath)

def _plotter(filepath, dpi=400):
    callback = (lambda fp: plt.savefig(fp,dpi=dpi))
    _write_nested(filepath,callback)

class GridDisplay:
    def __init__(self, num_items, nrows, ncols=4, rowsize=4, colsize=4):
        if nrows == -1:
            nrows = num_items//ncols +1 if num_items%ncols else num_items//ncols
        if nrows * ncols < num_items:
            raise ValueError(f"Cannot fit {num_items} items in a ({nrows},{ncols}) shape")

        self.nrows, self.ncols, self.k = nrows, ncols, 1
        #print(ncols,nrows, ncols*colsize, nrows*rowsize)
        self.fig = plt.figure(figsize=(colsize*ncols,rowsize*nrows))

    def add_plot(self,callback):
        ''' assumes callback takes axis as argument and does the
            necessary ops to put the correct data, titles, etc
        '''
        ax = plt.subplot(self.nrows,self.ncols,self.k) ; self.k = self.k+1
        callback(ax)
        plt.xticks([], []); plt.yticks([], [])
        plt.tight_layout()
        self.last_ax = ax

    def savefig(self,filepath, dpi=400):
        _plotter(filepath, dpi=dpi)

def _generate_gif(filepath, data, callback, num_frames, dpi=100, fps=1):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    images = [_plot_frame(i, data, callback, dpi=dpi) for i in range(num_frames)]
    imageio.mimsave(filepath, images, fps=fps)

def _plot_frame(i,data, callback, dpi, figsize=(10,5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # callback does the plotting job using data on ax at iteration i
    callback(i,ax,data)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def get_mode_hist(x,nbins=-1):
    ''' approximate the mode as the argmax  of binned frequencies'''
    if nbins == -1:
        nbins = max(10, len(x)//100 + 1)
    n, bins = np.histogram(x, nbins, density=True)

    idx_max = argrelmax(n)
    approx_x_pos = np.linspace(np.min(x), np.max(x), num=nbins)
    y_max = n[idx_max] ; x_max = approx_x_pos[idx_max]
    i_mode = np.argmax(y_max)

    return x_max[i_mode], y_max[i_mode]

def _basic_univar_distplot(data, ax=None, mode=True, kde=True,
                           nbins=None, distname=""):
    ''' a basic univariate distribution plot with mean and mode
        as verticale axes. distname is a short string to add to describe dist
    '''
    if ax is None:
        ax = plt
        if kde:
            sns.kdeplot(data, label=f"{distname} KDE") # avoid giving plt as ax to sns
    if isinstance(nbins,int):
        pass
    elif nbins in ['auto', 'sturges', 'fd', 'doane', 'scott', 'rice','sqrt']:
        pass
    else:
        nbins = 2*len(data)//50
    ax.hist(data, bins=nbins, alpha=0.4,
            color='royalblue', density=True, label="Histogram")
    ax.axvline(data.mean(), color='r', linestyle='--',
               linewidth=2, label=f'{distname} Mean')
    if mode:
        ax.axvline(get_mode_hist(data)[0], color='orange', linestyle='--',
                   linewidth=2, label=f'{distname} Mode')
    plt.legend()

def _compare_two_distplot(data1, data2, ax=None, mode=True,
                          kde=True, nbins=None, distnames=["",""]):
    ''' a basic univariate distribution plot with mean and mode
        as verticale axes. distname is a short string to add to describe dist
    '''
    assert len(distnames)==2
    if ax is None:
        ax = plt
        if kde:
            sns.kdeplot(data1, label=f"{distnames[0]} KDE", color='b') # avoid giving plt as ax to sns
            sns.kdeplot(data2, label=f"{distnames[1]} KDE", color='r')

    if isinstance(nbins,int):
        nbins1, nbins2 = nbins, nbins
    elif nbins in ['auto', 'sturges', 'fd', 'doane', 'scott', 'rice','sqrt']:
        nbins1, nbins2 = nbins, nbins
    else:
        nbins1, nbins2 = 2*len(data1)//50, 2*len(data2)//50

    ax.hist(data1, bins=nbins1, alpha=0.4,
            color='royalblue', density=True, label=f"{distnames[0]} Histogram")
    ax.axvline(data1.mean(), color='steelblue', linestyle='--',
               linewidth=2, label=f'{distnames[0]} Mean')

    ax.hist(data2, bins=nbins2, alpha=0.4,
            color='indianred', density=True, label=f"{distnames[1]} Histogram")
    ax.axvline(data2.mean(), color='tomato', linestyle='--',
               linewidth=2, label=f'{distnames[1]} Mean')
    if mode:
        ax.axvline(get_mode_hist(data1)[0], color='navy', linestyle='--',
                   linewidth=2, label=f'{distnames[0]} Mode')
        ax.axvline(get_mode_hist(data2)[0], color='darkred', linestyle='--',
                   linewidth=2, label=f'{distnames[1]} Mode')

def _mult_reduce(intlist):
    res = 1;
    for i in intlist:
        res *= i
    return res

def find_names(obj):
    frame = inspect.currentframe()
    for frame in iter(lambda: frame.f_back, None):
        frame.f_locals
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names

def mem_report(threshold=10):
    # finds all leaky tensors on gpu
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if _mult_reduce(obj.size())>threshold:
                print(type(obj), obj.size(),  find_names(obj))
