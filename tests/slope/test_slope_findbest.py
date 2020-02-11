import torch
import numpy as np
import matplotlib.pyplot as plt
import pprint as ppr

from causal.slope.slope import SlopeFunction, _function_to_index, _index_to_function
from causal.slope.utilities import (_get_dtype, _nan_to_zero, _parameter_score,
                                    _bin_int_as_array, _gaussian_score_emp_sse,
                                    _set_resolution)
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

pp = ppr.PrettyPrinter(indent=4)
#x,y,e = _torch_data(pows=[1,2])
x,y,e = _torch_data(pows=[2,5])

nofc = 13

slope_f = SlopeFunction(num_functions=nofc)
