import torch
import numpy as np
import matplotlib.pyplot as plt
import pprint as ppr

from causal.slope.slope import SlopeFunction, _function_to_index, _index_to_function
from causal.slope.utilities import (_get_dtype, _nan_to_zero, _parameter_score,
                                    _bin_int_as_array, _gaussian_score_emp_sse,
                                    _set_resolution)
from functions.miscellanea import _write_nested, _plotter, GridDisplay

SEED = 1020
torch.manual_seed(SEED)
np.random.seed(SEED)

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
x,y,e = _torch_data(pows=[1,2])
#x,y,e = _torch_data(pows=[2,5])

nofc = 13

slope_f = SlopeFunction(num_functions=nofc)

print(slope_f._design_matrix(x).shape)

print(_function_to_index(num_functions=nofc))
print(slope_f._nan_funcs, slope_f._nan_funcs_str, slope_f._nfuncs)

pp.pprint(slope_f._fit_generic(x,y+e))

for i,p in enumerate(slope_f._params):
    print(f'Coefficient for param #{i} corresponding to function {_index_to_function(num_functions=nofc)[i]} : {p}')

print(_nan_to_zero(slope_f._params))
print(f'parameter complexity scoring of generic fit {_parameter_score(slope_f._params)}')

func_name = _index_to_function(slope_f._nfuncs, del_nan=slope_f._isnan)

plt.plot(x.sort().values,y[x.sort().indices], 'k--')
plt.scatter(x,y+e,facecolor='none', edgecolor='r', alpha=0.5)
y_slope = slope_f._forward(x)
plt.plot(x.sort().values,y_slope[x.sort().indices], 'b-.')
plt.show()


for i in range(1,slope_f._nfuncs):
    print(f'\n summary for fit #{i}, name:{func_name[i]}')
    pp.pprint(slope_f._fit_index(x,y+e,i))
    print(f'--- end {i} ---')

score_list = []
score_list_str = []
for i in range(1,2**(slope_f._nfuncs)):

    bool_idx = _bin_int_as_array(i,slope_f._nfuncs)
    _res_mixed = slope_f._fit_mixed(x,y+e,bool_idx)

    print(f'\n summary for mixed-fit #{i}, name:{"+".join(_res_mixed["str_idx"])}')
    pp.pprint(slope_f._last_mixed_params)
    pp.pprint(_res_mixed)

    resolution = _set_resolution(x)
    score = _gaussian_score_emp_sse(_res_mixed['sse'], len(x), resolution=resolution) + _res_mixed['model_score']
    score_list.append(score) ; score_list_str.append('+'.join(_res_mixed['str_idx']))

score_list = np.array(score_list).ravel()
idx = np.argsort(score_list)
print(f'The increasing ordering for model complexity using {nofc} functions is:')
pp.pprint(list(np.array(score_list_str)[idx]))
print("corresponding to values:")
for e in score_list[idx]:
    print(e)
