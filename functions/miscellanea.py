''' Random functions useful for file management, display, etc.'''
import pickle
from pathlib import Path
import argparse, os


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
