import numpy as np
import torch
import numbers
from cdt.data import load_dataset
from sklearn.metrics import mean_squared_error
# have global values
RESOLUTION = 1e-02
data , labels = load_dataset('tuebingen',shuffle=False)
labels = labels.values
complain = "Datatype not supported, try Torch.tensors or np.ndarrays"

def _log(x):
    if isinstance(x,torch.Tensor):
        x[x!=0]= torch.log2(x[x!=0])
    elif isinstance(x,np.ndarray):
        idx = np.where(x!=0)
        x[idx] = np.log(x[idx])
    elif isinstance(x,numbers.Real):
        if x:
            x = np.log(x)
        else:
            pass
    else:
        raise NotImplementedError(complain,type(x))

    return x

def _read_index_tcep(i):
    return data.iloc[i].values

def _norm_x(x,norm_val):
    if isinstance(x,torch.Tensor):
        if x.max() == x.min():
            return norm_val*torch.ones(x.shape)
        else:
            return (x - x.min()) / (x.max()-x.min()) * norm_val
    elif isinstance(x,np.ndarray):
        if x.max() == x.min():
            return norm_val*np.ones(x.shape)
        else:
            return (x - x.min()) / (x.max()-x.min()) * norm_val
    elif isinstance(x,numbers.Real):
        return x
    else:
        raise NotImplementedError(complain,type(x))

def _nan_to_zero(x):
    if isinstance(x,torch.Tensor):
        x[torch.isnan(x)] = 0
    elif isinstance(x,np.ndarray):
        idx = np.where(np.isnan(x))
        x[idx] = 0
    elif isinstance(x,numbers.Real):
        if np.isnan(x):
            return 0
        else:
            return x
    else:
        raise NotImplementedError(complain,type(x))
    return x

def _log2_factorial(n,dtype='torch'):
    if dtype == 'torch':
        log2fact = _log(torch.arange(2,n+1)).sum()
    elif dtype =='numpy':
        log2fact = _log(np.arange(2,n+1)).sum()
    else:
        raise NotImplementedError(complain,dtype)
    return i_vals

def _log2_nCk(n,k, dtype='torch'):
    if k > n or k == 0:
        if dtype=='torch':
            return torch.Tensor([0])
        elif dtype=='numpy':
            return numpy.array([0])
        else:
            raise NotImplementedError(complain,dtype)
    else:
        return _log2_factorial(n,dtype=dtype) - _log2_factorial(k,dtype=dtype) - _log2_factorial(n-k,dtype=dtype)

def _log_n(z):
    if isinstance(x,torch.Tensor):
        z = torch.ceil(z)
        if z < 1:
            return z*0
        else:
            log_star = _log(z)
            summand = log_star
            while log_star > 0:
                log_star = _log(log_star)
                summand += log_star
        return summand + _log(torch.Tensor([2.865064]))

    elif isinstance(x,np.ndarray):
        z = np.ceil(z)
        if z < 1:
            return z*0
        else:
            log_star = _log(z)
            summand = log_star
            while log_star > 0:
                log_star = _log(log_star)
                summand += log_star
        return summand + _log(np.array([2.865064]))

    elif isinstance(x,numbers.Real):
        z = np.ceil(z)
        if z < 1:
            return z*0
        else:
            log_star = _log(z)
            summand = log_star
            while log_star > 0:
                log_star = _log(log_star)
                summand += log_star
        return summand + _log(2.865064)

    else:
        raise NotImplementedError(complain,type(x))

def _get_dtype(x):
    if isinstance(x,torch.Tensor):
        return 'torch'
    elif isinstance(x,np.ndarray):
        return 'numpy'
    else:
        raise ValueError("Either torch tensors or numpy arrays",type(x))

def _rand_perm(n,dtype='torch'):
    if dtype=='torch':
        return torch.randperm(n)
    elif dtype=='numpy':
        return np.random.permutation(n)
    else:
        raise NotImplementedError(complain,dtype)

def _mean_sq_err(x,y,func):
    if isinstance(x,torch.Tensor):
        torch.nn.MSELoss(reduction='sum')(y,func(x))
    elif isinstance(x,np.ndarray):
        return mean_squared_error(y,func(x))
    else:
        raise ValueError(complain,type(x),type(y))

def _fit_test_WMW(y,x,cycles=100, alpha=0.05):
    dtype = _get_dtype(x)
    assert dtype == _get_dtype(y)
    torch.manual_seed(1234)
    np.random.seed(1234)
    sig_count = 0
    for i in range(cycles):
        # 'universally' len(X) is nrows
        rand_perm = _rand_perm(n=len(X), dtype=dtype)
        split_idx = np.ceil(len(x)/2.0)
        x_tr, y_tr, x_te, y_te = x[rand_perm][:split_idx], y[rand_perm][:split_idx],\
                                 x[rand_perm][split_idx:], x[rand_perm][split_idx:]
        # possibly a basic func object to pass
        fit_result = _find_best_fit(y_tr,x_tr, function=_func)
        pass
        # NOT FINISHED
    pass

def _get_duplicate_positions(x):
    last = x[1]
    pass
    # NOT FINISHED

def _sort(x):
    if isinstance(x,torch.Tensor):
        return x.sort().values
    elif isinstance(x,np.ndarray):
        return np.sort(x)
    elif isinstance(x,numbers.Real):
        return x
    else:
        raise ValueError(complain, type(x))

def _roll(x,shift):
    ''' only rolls along first axis if several'''
    if isinstance(x,torch.Tensor):
        return torch.cat((x[-shift:], x[:-shift]))
    elif isinstance(x,np.ndarray):
        return np.roll(x,shift)
    elif isinstance(x,numbers.Real):
        return x
    else:
        raise ValueError(complain, type(x))

def _min_diff(x):
    x_s = _sort(x)
    delta = 1e-02
    if isinstance(x,torch.Tensor):
        new_delta = (x - _roll(x,1)).abs()[1:].min()
        delta = min(delta, new_delta) if new_delta>0 else delta
    elif isinstance(x,np.ndarray):
        new_delta = np.abs(x - _roll(x,1))[1:].min()
        delta = min(delta, new_delta) if new_delta>0 else delta
    else:
        raise ValueError(complain, type(x))

    return delta

def _data_precision(x):
    # np.round is universal, but on CPU
    # in doubt use types to do stuff
    precision = 1.0
    if isinstance(x,torch.Tensor):
        precision = torch.Tensor([precision])
        x = x[torch.round(x)!=x]
        while(len(x)) > 0:
            precision *= 0.1
            x *= 10
            x = x[torch.round(x)!=x]
    elif isinstance(x,np.ndarray):
        precision = np.array([precision])
        x = x[np.where(np.round(x)!=x)]
        while(len(x)) > 0:
            precision *= 0.1
            x *= 10
            x = x[np.where(np.round(x)!=x)]
    else:
        raise ValueError(complain, type(x))

    return precision


def _set_resolution(x, resolution=RESOLUTION, method='mindiff'):
    if method == 'mindiff':
        return _min_diff(x)
    else:
        return resolution
