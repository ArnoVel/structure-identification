import torch
import numpy as np
from torch.nn.functional import softmax, log_softmax

from fitting.gmm_fit import GaussianMixture
from fitting.data_examples import exp_gauss, unif, tri, two_dists_mixed
from functions.miscellanea import _write_nested
from matplotlib import pyplot as plt
import seaborn as sns

SEED = 12
torch.manual_seed(SEED)
np.random.seed(SEED)

def plotter_(filepath):
    callback = (lambda fp: plt.savefig(fp,dpi=100))
    _write_nested(filepath,callback)

# check cuda
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# set dimension
dim = 1
N = 10000  # Number of samples
# data
def data(N,dim=2):
    if dim==2:

        t = torch.linspace(0, 2 * np.pi, N + 1)[:-1]
        x = torch.stack((.5 + .4 * (t / 7) * t.cos(), .5 + .3 * t.sin()), 1)
        x = x + .02 * torch.randn(x.shape)
    elif dim==1:
        x = two_dists_mixed(N,sampler=unif,mus=[-0.1,0.3])
        x = torch.from_numpy(x).view(-1,1)

    return x

data = data(N,dim=dim)
# useful for 1d
low,up = data.min(), data.max()
x = data.type(dtype)

# model

model = GaussianMixture(30,sparsity=1, D=dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-01)

num_iters = 500

loss = np.zeros(num_iters)

for iter in range(num_iters):
    optimizer.zero_grad()  # Reset the gradients (PyTorch syntax...).
    cost = model.neglog_likelihood(x)  # Cost to minimize.
    #print(cost)
    cost.backward()  # Backpropagate to compute the gradient.
    optimizer.step()

    loss[iter] = cost.data.cpu().numpy()

    # sphinx_gallery_thumbnail_number = 6
    if iter in [0, 10, 50, 100, 150, 250, 350, 499]:
        plt.pause(.05)
        plt.figure(figsize=(8,8))
        model.plot(x)
        plt.title('Density, iteration ' + str(iter), fontsize=20)
        #plt.axis("equal")
        if dim==2:
            plt.axis([0,1,0,1])
        elif dim==1:
            vals = data.numpy().ravel()
            plt.hist(vals, bins=2*N//50, alpha=0.4,
                    color='royalblue', density=True, label="True Histogram")
            sns.kdeplot(vals, label="KDE ")
            plt.axis([low,up,0,1.5])
            plt.legend()
        plt.tight_layout() ; plt.pause(.05)
        plotter_(f'./tests/data/fitting/gmm_fit_iter{iter}')

qualifying_weights = (lambda t: [w for w in softmax(model.w, 0).detach().cpu().numpy() if w>t])
ref = qualifying_weights(1e-04); avg = sum(ref)/len(ref)
counts, weights, relweights = [1 for w in qualifying_weights(1e-04)],\
                              [w for w in qualifying_weights(1e-04)],\
                              [np.sqrt(w)/avg for w in qualifying_weights(1e-04)]
print(counts,weights)
print(sum(counts), sum(relweights))
