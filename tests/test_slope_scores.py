import torch
import numpy as np
import matplotlib.pyplot as plt
from causal.slope.slope import SlopeFunction, _function_to_index, _index_to_function
from causal.slope.utilities import _get_dtype, _nan_to_zero, _parameter_score
from functions.miscellanea import _write_nested, _plotter, GridDisplay

def _torch_data(n=700, pows=[2,3]):
    x,e = torch.normal(0,1,(500,)),\
            torch.normal(0,1,(500,))
    y = torch.sin(x)*torch.sigmoid(x)*((x-1).pow(pows[0])/5 - (x+1).pow(pows[1])/5)
    return x,y,e

def _np_data(n=700,pows=[2,3]):
    sigmoid = lambda x : (1 / (1 + np.exp(-x)))
    x,e = np.random.normal(0,1,(500,)),\
          np.random.normal(0,1,(500,))
    y = np.sin(x)*sigmoid(x)*(np.power(x-1,pows[0])/5 - np.power(x+1, pows[1])/5)
    return x,y,e

#x,y,e = _torch_data(pows=[1,2])
x,y,e = _torch_data(pows=[2,5])

slope_f = SlopeFunction(num_functions=13)

print(slope_f._design_matrix(x).shape)

print(_function_to_index(num_functions=13))
print(slope_f._nan_funcs, slope_f._nan_funcs_str, slope_f._nfuncs)

slope_f._fit_lstsq(x,y+e)
print(slope_f._params.shape)

for i,p in enumerate(slope_f._params):
    print(f'Coefficient for param #{i} corresponding to function {_index_to_function(num_functions=13)[i]} : {p}')

print(_nan_to_zero(slope_f._params))
print(_parameter_score(slope_f))
