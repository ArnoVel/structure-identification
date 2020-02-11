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
    x,e = torch.normal(0,1,(n,)),\
            torch.normal(0,1,(n,))
    y = torch.sin(x)*torch.sigmoid(x)*((x-1).pow(pows[0])/5 - (x+1).pow(pows[1])/5)

    return x,y,e

def _np_data(n=700,pows=[2,3]):
    sigmoid = lambda x : (1 / (1 + np.exp(-x)))
    x,e = np.random.normal(0,1,(n,)),\
          np.random.normal(0,1,(n,))
    y = np.sin(x)*sigmoid(x)*(np.power(x-1,pows[0])/5 - np.power(x+1, pows[1])/5)
    return x,y,e

pp = ppr.PrettyPrinter(indent=4)
#x,y,e = _torch_data(pows=[1,2])
x,y,e = _torch_data(pows=[2,3])
e = 1.5*e # try to have higher noise levels
y_n = y+e ; y_n = (y_n - y_n.mean(0)) / y_n.std(0)

y_disp = (y-y.mean(0)) / y.std(0)
SEED = 1020
torch.manual_seed(SEED)
np.random.seed(SEED)

nofc = 13

slope_f = SlopeFunction(num_functions=nofc)


_res_best_i = slope_f._find_best_fit_index(x,y_n)

pp.pprint(_res_best_i)

_res_best_mixed = slope_f._find_best_mixed_fit(x,y_n)

pp.pprint(_res_best_mixed)

plt.plot(x.sort().values,y_disp[x.sort().indices], 'k--', label='True function')
plt.scatter(x,y_n,facecolor='none', edgecolor='r', alpha=0.5)
_res_generic = slope_f._fit_generic(x,y_n)
y_slope = slope_f._forward(x) ; pp.pprint(_res_generic)
plt.plot(x.sort().values,y_slope[x.sort().indices], 'b-.', label='Generic Fit')
y_best_i = slope_f._forward_index(x,_res_best_i['index'])
plt.plot(x.sort().values,y_best_i[x.sort().indices], color='fuchsia', label=f'Best Index Fit (i={_res_best_i["index"]})')
# need to re fit correct mixed index, as we discard them as we iterate over them
check = slope_f._fit_mixed(x,y_n,_res_best_mixed['bool_idx'])
y_best_mixed = slope_f._forward_mixed(x,_res_best_mixed['bool_idx'])

plt.plot(x.sort().values,y_best_mixed[x.sort().indices], color='coral', label=f'Best Mixed Fit ({_res_best_mixed["str_idx"]})')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True)
plt.savefig('best_fits.png', dpi=120, bbox_inches="tight")
plt.show()

pp.pprint(check) ; pp.pprint(_res_best_mixed)
