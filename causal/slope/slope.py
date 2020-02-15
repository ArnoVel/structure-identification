import numpy as np
import torch
from .utilities import (_set_resolution, _nan_to_zero,
                        _sum_sq_err, _parameter_score,
                        _bin_int_as_array, _gaussian_score_emp_sse,
                        _unique)

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
        ''' warning: calling this with del_nan=True will reset
            the _marginal_params list to 0 and change its size
        '''
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
                self._isnan = torch.isnan(self._X.sum())
                nan_rows = torch.isnan(self._X.sum(1))
                self._nan_funcs = [i for i,b in enumerate(nan_rows) if b]
                self._nan_funcs_str = [_index_to_function(self._nfuncs)[nf] for nf in self._nan_funcs]
                self._X = self._X[~nan_rows]
                self._params = [p for i,p in enumerate(self._params) if not i in self._nan_funcs]
                self._nfuncs -= len(self._nan_funcs)
                self._marginal_params = [[0,0] for _ in range(self._nfuncs)]
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
                self._isnan = np.isnan(self._X.sum())
                self._nan_funcs = list(np.array(np.where(np.isnan(self._X.sum(1)))).ravel())
                self._nan_funcs_str = [_index_to_function(self._nfuncs)[nf] for nf in self._nan_funcs]
                not_nan_rows = np.where(~np.isnan(self._X.sum(1)))
                self._X = self._X[not_nan_rows]
                self._params = [p for i,p in enumerate(self._params) if not i in self._nan_funcs]
                self._nfuncs -= len(self._nan_funcs)
            else:
                # if dont check nan, put false
                self._isnan = False
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

    def _forward_mixed(self,x, bool_idx):
        ''' supposes a mixed fit has been done before,
            which matches with the current `bool_idx`.
        '''
        if not hasattr(self, '_last_mixed_params'):
            raise NotImplementedError("Cannot forward without fitting  mixed model first")

        assert any(bool_idx == self._last_mixed_params['bool_idx'])

        if not hasattr(self,'_X'):
            self._design_matrix(x)

        mask = (bool_idx == 1)
        if isinstance(x,torch.Tensor):
            _param_vec = torch.Tensor(self._last_mixed_params['params']).view(-1,1)
            _X_mask = self._X.t()[mask].t()
        elif isinstance(x,np.ndarray):
            _param_vec = np.array(self._last_mixed_params['params']).reshape(-1,1)
            _X_mask = self._X.T[mask].T
        # works in both torch & numpy, even if slower in numpy
        return (_X_mask @ _param_vec).flatten()

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
            sol, residuals, rank, singular_vals = np.linalg.lstsq(_X_i,y.reshape(-1,1), rcond=None)
            self._marginal_params[i] = sol
        else:
            raise ValueError(complain,type(x), type(y))

        # return a summary of the fit
        preds = self._forward_index(x,i)
        _sse = _sum_sq_err(y,preds)

        return {   'sse': _sse,
                    'model_score': _parameter_score(self._marginal_params[i]),
                    'params': self._marginal_params[i],
                    'index':i
                }
        # in the original, 0 maps to poly0, 1 to poly1..
        # here we follow a different order which can be
        # found above in the code, or using `_index_to_function`

    def _fit_mixed(self,x,y,bool_idx):
        ''' Fits a model by only using some basis functions,
            specified by an array with O's and 1's.
        '''
        mask = (bool_idx == 1) ; num_funcs = int(bool_idx.sum())

        if not hasattr(self,'_X'):
            self._design_matrix(x)
        if len(bool_idx) > self._nfuncs or len(bool_idx) < 1:
            return None
        assert type(x)==type(y)

        if isinstance(x,torch.Tensor):
            # by design, returns _X\y as [max(m,n),k], with _X [m,n] and y [m,k]
            # instead of _X\y as [n,k]. when m > n, fills the m-n remaining with RSS
            _X_mask = self._X.t()[mask].t()
            _params = torch.lstsq(y.view(-1,1),_X_mask).solution[:num_funcs]
        elif isinstance(x,np.ndarray):
            _X_mask = self._X.T[mask].T
            sol, residuals, rank, singular_vals = np.linalg.lstsq(_X_mask,y.reshape(-1,1), rcond=None)
            _params = sol
        else:
            raise ValueError(complain,type(x), type(y))

        str_func = _index_to_function(num_functions=self._nfuncs, del_nan=self._isnan)
        self._last_mixed_params = {   'params': _params,
                                      'bool_idx': bool_idx,
                                      'str_idx': [str_func[i] for i in range(self._nfuncs) if bool_idx[i]==1],
                                      }
        # return a summary of the fit
        preds = self._forward_mixed(x,bool_idx)
        _sse = _sum_sq_err(y,preds)

        return {    'sse': _sse,
                    'model_score': _parameter_score(self._last_mixed_params['params']),
                    'params': self._last_mixed_params['params'],
                    'bool_idx': bool_idx,
                    'str_idx': [str_func[i] for i in range(self._nfuncs) if bool_idx[i]==1],
                }
        # in the original, 0 maps to poly0, 1 to poly1..
        # here we follow a different order which can be
        # found above in the code, or using `_index_to_function`

    def _fit_generic(self,x,y):
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

        preds = self._forward(x)
        _sse = _sum_sq_err(y,preds)
        return  {   'sse': _sse,
                    'model_score': _parameter_score(self._params),
                    'params': self._params
                }

    def _find_best_fit_index(self,x,y):
        if not hasattr(self,'_X'):
            self._design_matrix(x, del_nan=True)
        if hasattr(self,'_isnan'):
            func_index = _function_to_index(self._nfuncs, del_nan=self._isnan)
        else:
            func_index = _function_to_index(self._nfuncs, del_nan=False)
        # start with linear fit
        rr = self._fit_index(x,y, func_index['poly1'])
        resolution = _set_resolution(x)
        score = _gaussian_score_emp_sse(rr['sse'], len(x), resolution=resolution) + rr['model_score']


        for i in range(1,self._nfuncs):
            # instead of leaving the nan option, just exclude the nan funcs,
            # when computing _design_matrix() above.
            rr_new = self._fit_index(x,y,i)
            score_new = _gaussian_score_emp_sse(rr_new['sse'], len(x), resolution=resolution) + rr_new['model_score']
            if score_new < score:
                score = score_new
                rr = rr_new
        return rr

    def _find_best_mixed_fit(self,x,y):
        ''' _fit_index only uses basis function #i.
            mixed fits are a number k < _nfuncs of basis functions,
            which yield best balance fit + complexity.
            The default tries all 2^(k)-1 combinations of basis functions
        '''
        if not hasattr(self,'_X'):
            self._design_matrix(x, del_nan=True)
        if hasattr(self,'_isnan'):
            func_index = _function_to_index(self._nfuncs, del_nan=self._isnan)
        else:
            func_index = _function_to_index(self._nfuncs, del_nan=False)

        # start with linear fit
        resolution = _set_resolution(x)
        rr = self._fit_index(x,y, func_index['poly1'])
        score = _gaussian_score_emp_sse(rr['sse'], len(x), resolution=resolution) + rr['model_score']

        for i in range(1,2**(self._nfuncs)):
            bool_idx = _bin_int_as_array(i,self._nfuncs)
            rr_new = self._fit_mixed(x,y,bool_idx)
            score_new =  _gaussian_score_emp_sse(rr_new['sse'], len(x), resolution=resolution) + rr_new['model_score']
            if score_new < score:
                score = score_new
                rr = rr_new
        return rr

    def _fit_wrapper(x,y, mixed=False):
        # rounding is same in np & torch
        min_num = 5e04 ; max_x = x.max() ; prec = 1e05
        x_f = (x*prec).round()
        unique, counts = _unique(x)
        mean_c, sd_c, len_x = counts.mean(), counts.std(), len(x)

        _r_bestfit = self._find_best_mixed_fit(x,y) if mixed else self._find_best_fit_index(x,y)
