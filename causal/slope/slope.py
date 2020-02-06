import numpy as np
import torch
from .utilities import _set_resolution, _nan_to_zero
from .utilities import *

NUM_FUNC_CLASSES = 7

def _function_to_index(num_functions, del_nan=False):
    if num_functions < 5:
        num_functions = 5
    elif num_functions > 13:
        num_functions = 13
    elif not isinstance(num_functions,int):
        raise ValueError("This argument should be (int)",type(num_functions))
    poly_n = (num_functions - 3) // 2 # nonzero powers:
    _pows = range(1,poly_n+1)
    if del_nan:
        # in case log returns nan values, delete the log basis function
        poly_n = (num_functions - 2) // 2 # nonzero powers:
        _pows = range(1,poly_n+1)
        return {    'exp':1,
                    'poly0':0,
                    **{f'poly{i}':1+i for i in _pows},
                    **{f'poly_inv{i}':1+poly_n+i for i in _pows}
                    }
    else:
        return {    'exp':1,
                    'log':2,
                    'poly0':0,
                    **{f'poly{i}':2+i for i in _pows},
                    **{f'poly_inv{i}':2+poly_n+i for i in _pows}
                    }

def _index_to_function(num_functions, del_nan=False):
    if num_functions < 5:
        num_functions = 5
    elif num_functions > 13:
        num_functions = 13
    elif not isinstance(num_functions,int):
        raise ValueError("This argument should be (int)",type(num_functions))
    poly_n = (num_functions - 3) // 2 # nonzero powers:
    _pows = range(1,poly_n+1)
    if del_nan:
        # in case log returns nan values, delete the log basis function
        poly_n = (num_functions - 2) // 2 # nonzero powers:
        _pows = range(1,poly_n+1)
        return {    1:'exp',
                    0:'poly0',
                    **{1+i:f'poly{i}' for i in _pows},
                    **{1+poly_n+i:f'poly_inv{i}' for i in _pows}
                    }
    else:
        return {    1:'exp',
                    0:'poly0',
                    2:'log',
                    **{2+i:f'poly{i}' for i in _pows},
                    **{2+poly_n+i:f'poly_inv{i}' for i in _pows}
                    }


class SlopeFunction:
    ''' class which mimics the fofx forward pass in the R slope code,
        to which we add the np/torch choice & a flexible poly & inv poly #'''
    def __init__(self,num_functions=NUM_FUNC_CLASSES ,init_params=-1):
        ''' consider the most basic is a + b*x + c/x + d*exp(x) + e*log(x),
            namely with only deg 1 poly, deg 1 inv poly.
            Avoid more than deg 5 in both +/- powers
        '''
        if num_functions < 5:
            num_functions = 5
        elif num_functions > 13:
            num_functions = 13
        elif not isinstance(num_functions,int):
            raise ValueError("This argument should be (int)",type(num_functions))

        poly_n = (num_functions - 3) // 2 # nonzero powers:

        self._nfuncs = 3 + 2*poly_n
        self._pows = range(1,poly_n+1)
        self._marginal_params = [[0,0] for _ in range(self._nfuncs)]
        if init_params==-1:
            self._params = [0 for _ in range(self._nfuncs)]
        else:
            raise NotImplementedError('don\'t set params please')

    def _design_matrix(self,x,del_nan=True):
        # returns f(x) in matrix form for all x's
        # setting del_nan=True is equivalent to the `fitGeneric` method
        assert len(self._params) == self._nfuncs
        resolution = _set_resolution(x)
        if isinstance(x,torch.Tensor):
            x_non_zero = x.clone()
            x_non_zero[x_non_zero==0] = resolution
            # the authors' way: accept the NaN values, and
            # use them in model selection to discard the basis function
            self._X = torch.stack([
                            torch.ones(x.shape),
                            x.exp(),
                            x_non_zero.log(),
                            *[
                                x.pow(i) for i in self._pows
                            ],
                            *[
                                x_non_zero.pow(-i) for i in self._pows
                            ]
                            ])
            if del_nan:
                nan_rows = torch.isnan(self._X.sum(1))
                self._nan_funcs = [i for i,b in enumerate(nan_rows) if b]
                self._nan_funcs_str = [_index_to_function(self._nfuncs)[nf] for nf in self._nan_funcs]
                self._X = self._X[~nan_rows]
                self._params = [p for i,p in enumerate(self._params) if not i in self._nan_funcs]
                self._nfuncs -= len(self._nan_funcs)
            self._X = self._X.t()

        elif isinstance(x,np.ndarray):
            x_non_zero = x.copy()
            x_non_zero[x_non_zero==0] = resolution
            # the authors' way: accept the NaN values, and
            # use them in model selection to discard the basis function
            self._X = np.vstack([
                            np.ones(x.shape),
                            np.exp(x),
                            np.log(x_non_zero),
                            *[
                                np.power(x,i) for i in self._pows
                            ],
                            *[
                                np.power(x_non_zero,-i) for i in self._pows
                            ]
                            ])
            if del_nan:
                self._nan_funcs = list(np.array(np.where(np.isnan(self._X.sum(1)))).ravel())
                self._nan_funcs_str = [_index_to_function(self._nfuncs)[nf] for nf in self._nan_funcs]
                not_nan_rows = np.where(~np.isnan(self._X.sum(1)))
                self._X = self._X[not_nan_rows]
                self._params = [p for i,p in enumerate(self._params) if not i in self._nan_funcs]
                self._nfuncs -= len(self._nan_funcs)
            self._X = self._X.T

        else:
            return ValueError(complain,type(x))

        return self._X

    def _forward(self,x):
        if not hasattr(self,'_X'):
            self._design_matrix(x)
            # if not learned params forward anyway with 0's ..
        if isinstance(x,torch.Tensor):
            _param_vec = torch.Tensor(self._params).view(-1,1)
        elif isinstance(x,np.ndarray):
            _param_vec = np.array(self._params).reshape(-1,1)
        # works in both torch & numpy, even if slower in numpy
        return (self._X @ _param_vec).flatten()

    def _forward_index(self,x,i):
        if not hasattr(self,'_X'):
            self._design_matrix(x)
        if isinstance(x,torch.Tensor):
            _param_vec = torch.Tensor(self._marginal_params[i]).view(-1,1)
            _X_i = self._X.t()[[0,i],:].t()
        elif isinstance(x,np.ndarray):
            _param_vec = np.array(self._marginal_params[i]).reshape(-1,1)
            _X_i = self._X.T[[0,i],:].T
        # works in both torch & numpy, even if slower in numpy
        return (_X_i @ _param_vec).flatten()

    def _fit_index(self,x,y,i):
        ''' similar to `fitI` in the original R code,
            instead of using all basis functions,
            only uses function #i + bias. Similar to a GLM.
        '''
        if not hasattr(self,'_X'):
            self._design_matrix(x)
        if i > self._nfuncs or i < 1:
            return None
        assert type(x)==type(y)

        if isinstance(x,torch.Tensor):
            # by design, returns _X\y as [max(m,n),k], with _X [m,n] and y [m,k]
            # instead of _X\y as [n,k]. when m > n, fills the m-n remaining with RSS
            _X_i = self._X.t()[[0,i],:].t()
            self._marginal_params[i] = torch.lstsq(y.view(-1,1),_X_i).solution[:2]
        elif isinstance(x,np.ndarray):
            _X_i = self._X.T[[0,i],:].T
            print(_X_i.shape)
            sol, residuals, rank, singular_vals = np.linalg.lstsq(_X_i,y.reshape(-1,1), rcond=None)
            self._marginal_params[i] = sol
        else:
            return ValueError(complain,type(x), type(y))
        # in the original, 0 maps to poly0, 1 to poly1..
        # here we follow a different order which can be
        # found above in the code, or using `_index_to_function`

    def _fit_lstsq(self,x,y):
        assert type(x)==type(y)
        if not hasattr(self,'_X'):
            self._design_matrix(x)
        if isinstance(x,torch.Tensor):
            # by design, returns _X\y as [max(m,n),k], with _X [m,n] and y [m,k]
            # instead of _X\y as [n,k]. when m > n, fills the m-n remaining with RSS
            self._params = torch.lstsq(y.view(-1,1),self._X).solution[:self._nfuncs]
        elif isinstance(x,np.ndarray):
            sol, residuals, rank, singular_vals = np.linalg.lstsq(self._X,y.reshape(-1,1), rcond=None)
            self._params = sol
        else:
            return ValueError(complain,type(x), type(y))

    def _find_best_fit(self,x,y):
        pass
