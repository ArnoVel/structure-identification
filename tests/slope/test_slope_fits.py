import torch
import numpy as np
import matplotlib.pyplot as plt
from causal.slope.slope import SlopeFunction, _function_to_index, _index_to_function
from causal.slope.utilities import (_get_dtype, _nan_to_zero, _parameter_score,
                                    _bin_int_as_array)
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
x,y,e = _torch_data(pows=[2,4])

def _plotter(x,y,e,slope_item=None,dtype='torch',check_all_i=False):
    if dtype=='torch':
        if check_all_i:
            for i in range(1,slope_item._nfuncs):
                plt.plot(x.sort().values,y[x.sort().indices], 'k--', lw=2)
                y_slope = slope_item._forward(x)
                plt.plot(x.sort().values,y_slope[x.sort().indices], 'b-.', lw=2)

                slope_item._fit_index(x,y+e,i)
                y_i = slope_item._forward_index(x,i)
                plt.plot(x.sort().values,y_i[x.sort().indices], 'g-1', lw=1.3)
                plt.scatter(x,y+e,facecolor='none', edgecolor='r')
                plt.show()
        else:
            plt.plot(x.sort().values,y[x.sort().indices], 'k--')
            plt.scatter(x,y+e,facecolor='none', edgecolor='r')
            if slope_item is not None:
                y_slope = slope_item._forward(x)
                plt.plot(x.sort().values,y_slope[x.sort().indices], 'b-.')
                plt.show()

    elif dtype=='numpy':
        if check_all_i:
            for i in range(1,slope_item._nfuncs):
                idx = np.argsort(x)
                plt.plot(x[idx],y[idx], 'k--',lw=2)
                y_slope = slope_item._forward(x)
                plt.plot(x[idx],y_slope[idx], 'b-.', lw=2)

                slope_item._fit_index(x,y+e,i)
                y_i = slope_item._forward_index(x,i)
                plt.plot(x[idx],y_i[idx], 'g-1', lw=1.3)
                plt.scatter(x,y+e,facecolor='none', edgecolor='r')
                plt.show()
        else:
            idx = np.argsort(x)
            plt.plot(x[idx],y[idx], 'k--', lw=2)
            if slope_item is not None:
                y_slope = slope_item._forward(x)
                plt.plot(x[idx],y_slope[idx], 'b-.')
            plt.scatter(x,y+e,facecolor='none', edgecolor='r')

            plt.show()

def _grid_plotter_onefunc(x,y,e,slope_item):
    assert type(x) == type(y+e)
    display = GridDisplay(num_items=slope_item._nfuncs, nrows=-1, ncols=3)
    if isinstance(x,torch.Tensor):
        for i in range(1,slope_item._nfuncs):
            print(f'func #{i}')
            def callback(ax,x,y,e,i):
                ax.plot(x.sort().values,y[x.sort().indices], 'k--', lw=2)
                y_slope = slope_item._forward(x)
                ax.plot(x.sort().values,y_slope[x.sort().indices], 'b-.', lw=2)

                slope_item._fit_index(x,y+e,i)
                y_i = slope_item._forward_index(x,i)
                ax.plot(x.sort().values,y_i[x.sort().indices], 'g-1', lw=1.3, alpha=0.7)
                ax.scatter(x,y+e,facecolor='none', edgecolor='r')
                ax.set_title(_index_to_function(slope_item._nfuncs, del_nan=True)[i])

            display.add_plot(callback=(lambda ax: callback(ax,x,y,e,i)))
    elif isinstance(x,np.ndarray):
        for i in range(1,slope_item._nfuncs):
            def callback(ax,x,y,e,i):
                idx = np.argsort(x)
                ax.plot(x[idx],y[idx], 'k--',lw=2)
                y_slope = slope_item._forward(x)
                ax.plot(x[idx],y_slope[idx], 'b-.', lw=2)

                slope_item._fit_index(x,y+e,i)
                y_i = slope_item._forward_index(x,i)
                ax.plot(x[idx],y_i[idx], 'g-1', lw=1.3, alpha=0.7)
                ax.scatter(x,y+e,facecolor='none', edgecolor='r')
                ax.set_title(_index_to_function(slope_item._nfuncs, del_nan=True)[i])

            display.add_plot(callback=(lambda ax: callback(ax,x,y,e,i)))
    display.fig.suptitle(r'Slope on ANM data $Y= f(X)+N$', fontsize=20)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _grid_plotter_mixed(x,y,e,slope_item):
    assert type(x) == type(y+e)

    combinations = range(1,2**(slope_f._nfuncs))
    display = GridDisplay(num_items=len(combinations), nrows=-1, ncols=8, rowsize=4, colsize=4)

    if isinstance(x,torch.Tensor):
        for i in range(1,2**(slope_f._nfuncs)):
            def callback(ax,x,y,e,i):
                bool_idx = _bin_int_as_array(i,slope_f._nfuncs)
                _res_mixed = slope_f._fit_mixed(x,y+e,bool_idx)

                ax.plot(x.sort().values,y[x.sort().indices], 'k--', lw=2) # true func
                y_slope = slope_item._forward(x)
                ax.plot(x.sort().values,y_slope[x.sort().indices], 'b-.', lw=2) # generic fit

                y_mixed = slope_item._forward_mixed(x,bool_idx)

                ax.plot(x.sort().values,y_mixed[x.sort().indices], 'g-1', lw=1.3, alpha=0.7)
                ax.scatter(x,y+e,facecolor='none', edgecolor='r')
                ax.set_title('+'.join(_res_mixed['str_idx']), fontsize=8)

            display.add_plot(callback=(lambda ax: callback(ax,x,y,e,i)))

    elif isinstance(x,np.ndarray):
        for i in range(1,2**(slope_f._nfuncs)):
            def callback(ax,x,y,e,i):
                bool_idx = _bin_int_as_array(i,slope_f._nfuncs)
                _res_mixed = slope_f._fit_mixed(x,y+e,bool_idx)

                idx = np.argsort(x)
                ax.plot(x[idx],y[idx], 'k--',lw=2)
                y_slope = slope_item._forward(x)
                ax.plot(x[idx],y_slope[idx], 'b-.', lw=2)

                y_mixed = slope_item._forward_mixed(x,bool_idx)

                ax.plot(x[idx],y_mixed[idx], 'g-1', lw=1.3, alpha=0.7)
                ax.scatter(x,y+e,facecolor='none', edgecolor='r', alpha=0.7)
                ax.set_title('+'.join(_res_mixed['str_idx']), fontsize=3)

            display.add_plot(callback=(lambda ax: callback(ax,x,y,e,i)))

    display.fig.suptitle(r'Slope-Mixed on ANM data $Y= f(X)+N$', fontsize=15)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
    plt.show()

nofc = 8
slope_f = SlopeFunction(num_functions=nofc)

print(slope_f._design_matrix(x).shape)

print(_function_to_index(num_functions=nofc))
print(slope_f._nan_funcs, slope_f._nan_funcs_str, slope_f._nfuncs)

slope_f._fit_generic(x,y+e)
print(slope_f._params.shape)
#_plotter(x,y,e,slope_item=slope_f,dtype=_get_dtype(x), check_all_i=True)
#_grid_plotter_onefunc(x,y,e,slope_f)
_grid_plotter_mixed(x,y,e,slope_f)
