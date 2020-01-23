''' Random functions useful for file management, display, etc.'''
import pickle
from pathlib import Path
import argparse, os, gc, inspect
import matplotlib.pyplot as plt
import torch

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
    def __init__(self, nrows, ncols=4, rowsize=4, colsize=4):
        self.nrows, self.ncols, self.k = nrows, ncols, 1
        plt.figure(figsize=(colsize*ncols,rowsize*nrows))

    def add_plot(self,callback):
        ''' assumes callback takes axis as argument and does the
            necessary ops to put the correct data, titles, etc
        '''
        ax = plt.subplot(self.nrows,self.ncols,self.k) ; self.k = self.k+1
        callback(ax)
        plt.xticks([], []); plt.yticks([], [])
        plt.tight_layout()

    def savefig(self,filepath, dpi=400):
        _plotter(filepath, dpi=dpi)


def _mult_reduce(intlist):
    res = 1;
    for i in intlist:
        res *= i
    return res

def find_names(obj):
    frame = inspect.currentframe()
    for frame in iter(lambda: frame.f_back, None):
        frame.f_locals
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names

def mem_report(threshold=10):
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if _mult_reduce(obj.size())>threshold:
                print(type(obj), obj.size(),  find_names(obj))
