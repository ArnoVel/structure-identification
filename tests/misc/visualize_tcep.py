from cdt.data import load_dataset
from sklearn.preprocessing import scale

from functions.tcep_utils import cut_num_pairs,ensemble_score, _get_wd
from functions.miscellanea import _write_nested, _plotter, GridDisplay
from matplotlib import pyplot as plt
import seaborn as sns

def callback(ax,x,y,name):
    plt.pause(.01)
    ax.scatter(x,y,s=15, alpha=0.4, c='k')
    #ax.set_title(name, fontsize=1)
    plt.axis([0,1,0,1])
    plt.xticks([], []); plt.yticks([], [])
    plt.tight_layout() ; plt.pause(.01)

def process(row):
    x,y = (scale(row['A']), scale(row['B']))
    return x,y

data , labels = load_dataset('tuebingen',shuffle=False)
labels = labels.values
cut_num_pairs(data,num_max=5000)
ncols=5
nrows = data.shape[0]//ncols+1  if data.shape[0]%ncols else data.shape[0]//ncols
display = GridDisplay(nrows=nrows, ncols=ncols, rowsize=4, colsize=10)

print(nrows, ncols)

for i,row in data.iterrows():
    x,y = process(row)
    display.add_plot(callback=(lambda ax: callback(ax,x,y,name=i)))

display.savefig(f'./tests/data/tcep/tcep_pairs')
