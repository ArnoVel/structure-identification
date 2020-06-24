import torch
import numpy as np
from torch.nn.functional import softmax, log_softmax
from functions.generators.generators import *

from fitting.gmm_fit import GaussianMixture
from fitting.data_examples import exp_gauss, unif, tri, two_dists_mixed
from functions.miscellanea import _write_nested, _plotter, GridDisplay
from matplotlib import pyplot as plt
import seaborn as sns

SEED = 12
torch.manual_seed(SEED)
np.random.seed(SEED)

# check cuda
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# set dimension
dim = 2
N = 1000  # Number of samples
# data
def data(N,dim=2):
    if dim==2:
        t = torch.linspace(0, 2 * np.pi, N + 1)[:-1]
        x = torch.stack((.5 + .4 * (t / 7) * t.cos(), .5 + .3 * t.sin()), 1)
        x = x + .02 * torch.randn(x.shape)
    elif dim==1:
        x = two_dists_mixed(N,sampler=tri, mus=[-0.2,0.1])
        x = torch.from_numpy(x).view(-1,1)
    return x

## ANM DATA, FOR NOW RESCALE TO (0,1) FOR DISPLAY (quite difficult to tune..)
def anm_data(N,dim=2):
    if dim==2:
        s = CauseSampler(sample_size=N)
        x = s.uniform()
        ms = MechanismSampler(x) ; mech = ms.tanhSum()
        x = torch.from_numpy(x).view(-1,1)
        x = (x-x.mean(0))/x.std(0); x = (x - x.min()) / (x.max() - x.min())

        y = mech(x.numpy()); y = torch.from_numpy(y).view(-1,1)
        delta_f = (y.max() - y.min()) * 0.1
        print(x.shape, y.shape, delta_f.shape)
        e = delta_f * torch.randn(x.shape) ; y_n = y+e

        y_n = (y_n-y_n.mean(0))/y_n.std(0)
        y_n = (y_n - y_n.min()) / (y_n.max() - y_n.min())
        x = torch.cat([x,y_n],1)

        return (x,mech)

    elif dim==1:
        x = two_dists_mixed(N,sampler=tri, mus=[-0.2,0.1])
        x = torch.from_numpy(x).view(-1,1)

        return (x, None, None)

#data = data(N,dim=dim)
data, mech = anm_data(N,dim=2)
anm_flag = True
# useful for 1d
low,up = data.min(), data.max()
x = data.type(dtype)

# model

model = GaussianMixture(30,sparsity=1, D=dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-01)

num_iters = 500

loss = np.zeros(num_iters)

# display utilities
display_its = [0, 10, 50, 100, 150, 250, 350, 499]
# default num cols is 4
display = GridDisplay(num_items=len(display_its), nrows=-1, ncols=3)

def callback(ax, iter, dim, low, up):
    plt.pause(.05)
    model.plot(x, ax=ax)
    ax.set_title('Density, iter ' + str(iter))
    #plt.axis("equal")
    if dim==2:
        if anm_flag:
            linsp = np.linspace(0,1,1000)
            y_linsp = mech(linsp)
            y_linsp = (y_linsp-y_linsp.mean(0))/y_linsp.std(0)
            y_linsp = (y_linsp - y_linsp.min()) / (y_linsp.max() - y_linsp.min())
            ax.plot(linsp, y_linsp, 'k--')
        if not anm_flag:
            plt.axis([0,1,0,1])
    elif dim==1:
        vals = data.numpy().ravel()
        ax.hist(vals, bins=2*N//50, alpha=0.4,
                color='royalblue', density=True, label="True Histogram")
        sns.kdeplot(vals, label="KDE ",ax=ax)
        plt.axis([low,up,0,2])
        plt.legend()
    plt.xticks([], []); plt.yticks([], [])
    plt.tight_layout() ; plt.pause(.05)

## Main training loop
print(x.shape)

for iter in range(num_iters):
    optimizer.zero_grad()  # Reset the gradients (PyTorch syntax...).
    cost = model.neglog_likelihood(x)  # Cost to minimize.
    #print(cost)
    cost.backward()  # Backpropagate to compute the gradient.
    optimizer.step()

    loss[iter] = cost.data.cpu().numpy()

    if iter in display_its:
        display.add_plot(callback=(lambda ax: callback(ax,
                                                    dim=dim,
                                                    iter=iter,
                                                    low=low,
                                                    up=up)))
    print(iter)

print('Loss over iterations:')
plt.show()
plt.plot(loss)
plt.show()
# display.savefig(f'./tests/data/fitting/gmm/dim-two/anm_ex_uniform_tanhsum{iter}')

# qualifying_weights = (lambda t: [w for w in softmax(model.w, 0).detach().cpu().numpy() if w>t])
# ref = qualifying_weights(1e-04); avg = sum(ref)/len(ref)
# counts, weights, relweights = [1 for w in qualifying_weights(1e-04)],\
#                               [w for w in qualifying_weights(1e-04)],\
#                               [np.sqrt(w)/avg for w in qualifying_weights(1e-04)]
# print(counts,weights)
# print(sum(counts), sum(relweights))
