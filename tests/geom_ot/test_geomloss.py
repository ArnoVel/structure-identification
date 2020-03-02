import numpy as np
import matplotlib.pyplot as plt
import time
from random import choices, choice, seed
from itertools import product
from imageio import imread
from matplotlib import pyplot as plt
import torch
from geomloss import SamplesLoss
from functions.generators.generators import *
from functions.miscellanea import _write_nested, _plotter, GridDisplay
from dependence import c2st
from causal.generative.geometric import CausalGenGeomNet,GenerativeGeomNet
from causal.slope.utilities import _log, _parameter_score

## this secures basic features of the GeomLoss package,
## and runs a few examples using geometric losses to train
## neural nets as conditional generative models for 2D distributions

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

print(f'Using cuda? {"yes" if use_cuda else "no"}, data type is then {dtype}')

def _score_wrapper(net):
    param_flat = torch.cat([p.detach().flatten() for p in net.parameters()])
    return _parameter_score(param_flat)

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
def anm_data(N,dim=2, SEED=1020):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    seed(SEED)
    causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
    base_noises = ['normal', 'student', 'triangular', 'uniform',
                   'beta', 'semicircular']
    mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']

    global c,bn,m
    c,bn,m = choice(causes), choice(base_noises), choice(mechanisms)
    print(f'random choice of ANM: {c,bn,m}')
    DtSpl = DatasetSampler(N=1, n=N, anm=True,
                           base_noise=bn,
                           cause_type=c,
                           mechanism_type=m,
                           with_labels=False)
    DtSpl.__iter__() ; x = torch.from_numpy(next(DtSpl)).t()

    return x

def display_samples(ax, x, color) :
    x_ = x.detach().cpu().numpy()
    ax.scatter( x_[:,0], x_[:,1], 25*500 / len(x_), color, edgecolors='none' )

def gradient_flow(loss, lr=.05, loss_info=None) :
    """Flows along the gradient of the cost function, using a simple Euler scheme.

    Parameters:
        loss ((x_i,y_j) -> torch float number):
            Real-valued loss function.
        lr (float, default = .05):
            Learning rate, i.e. time step.
    """

    # Parameters for the gradient descent
    total_ = 5 ; num_displays = 15
    Nsteps = int(total_/lr)+1 # base was 5/lr
    display_its = [int(t/lr) for t in torch.linspace(0,total_,num_displays)]
    print(f'display its: {display_its}')
    # Use colors to identify the particles
    colors = (10*X_i[:,0]).cos() * (10*X_i[:,1]).cos()
    colors = colors.detach().cpu().numpy()

    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()

    # We're going to perform gradient descent on Loss(α, β)
    # wrt. the positions x_i of the diracs masses that make up α:
    x_i.requires_grad = True
    # try using Adam
    optim = torch.optim.Adam([x_i], lr=lr)

    t_0 = time.time()
    display = GridDisplay(num_items=num_displays, nrows=-1, ncols=5)
    for i in range(Nsteps): # Euler scheme ===============
        # Compute cost and gradient
        L_αβ = loss(x_i, y_j)
        optim.zero_grad()
        L_αβ.backward()
        #[g]  = torch.autograd.grad(L_αβ, [x_i])
        if i in display_its : # display
            print(f'loop iter {i}')
            #print(f'check gradient and loss magnitudes: {g.norm().data, L_αβ.data}')
            #print(f'check gradient and loss magnitudes: {x_i.grad.data, L_αβ.data}')
            def callback(ax, x_i, y_j, colors):
                plt.set_cmap("hsv")
                display_samples(ax, y_j, [(.55,.55,.95)])
                display_samples(ax, x_i, colors)
                ax.set_title("t = {:1.2f}".format(lr*i))

                #plt.axis([0,1,0,1])
                #plt.gca().set_aspect('equal', adjustable='box')
                plt.xticks([], []); plt.yticks([], [])
                plt.tight_layout()
            display.add_plot(callback=(lambda ax: callback(ax, x_i, y_j, colors)))


        # in-place modification of the tensor's values
        #x_i.data -= lr * len(x_i) * g
        optim.step()
    display.fig.suptitle((f"Gradient flows with loss {loss_info};"+
                          f"\n T = {lr*i}, elapsed time: {int(1e03*(time.time() - t_0)/Nsteps)/1e03}s/it"+
                          f"\n on ANM data {c,m,bn}"),
                          fontsize=10)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.93])


def two_dims_fit():
    N, M = (100, 100) if not use_cuda else (3000, 3000) # if not same # pts for D1 D2
    C = torch.rand(2,2) ; C = C.t() @ C
    Y_j = anm_data(N, dim=2, SEED=1612) #Y_j = data(N, dim=2)
    X_i = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.randn(2)+1, covariance_matrix=C+torch.eye(len(C))).sample((N,))
    X_i = (X_i - X_i.mean(0)) / X_i.std(0) / 2
    X_i, Y_j = X_i.type(dtype), Y_j.type(dtype)
    print(X_i.shape, Y_j.shape)

    # rbf kernel gradient flow
    loss_info = {"loss":"sinkhorn", "p":2, "blur":1e-01}
    gradient_flow( SamplesLoss(**loss_info) , lr=5e-02, loss_info=loss_info)
    plt.show()

def test_geom_nets(XY):
    XY = XY.type(dtype); X,Y = XY[:,0].clone(), XY[:,1].clone()
    causal_geom_net = CausalGenGeomNet(loss="sinkhorn", p=1)
    causal_geom_net.set_data(X,Y)
    causal_geom_net.fit_two_directions()

    for t in ["mmd-gamma","c2st-nn","c2st-knn"]:
        data_prob = causal_geom_net.data_probability(test_type=t, num_tests=5)
        print(f'obtained -log(likelihood) for test {t}: causal {-_log(data_prob[0])} vs. anticausal {-_log(data_prob[1])}')
    test_losses = causal_geom_net.test_loss()
    print(f'test loss bits with exp likelihood: causal {-_log(torch.exp(-test_losses[0]))} vs. anticausal {-_log(torch.exp(-test_losses[1]))}')
    print(f'parameter compression: X --> Y {_score_wrapper(causal_geom_net._fcm_net_causal)}')
    print(f'parameter compression: Y --> X {_score_wrapper(causal_geom_net._fcm_net_anticausal)}')

def _sample_anms_geom_nets(N=5):
    causes = ['gmm', 'subgmm','supgmm','subsupgmm','uniform','mixtunif']
    base_noises = ['normal', 'student', 'triangular', 'uniform',
                   'beta', 'semicircular']
    mechanisms = ['spline','sigmoidam','tanhsum','rbfgp']

    for c,m,bn in product(causes, mechanisms, base_noises):
        print(f'testing GeomLosses on ANM data with N={N} datasets')
        print(f'the ANM structure is (cause,mechanism,noise) = {c,m,bn}')
        DtSpl = DatasetSampler(N=N, n=1000, anm=True,
                               base_noise=bn,
                               cause_type=c,
                               mechanism_type=m,
                               with_labels=False)
        t_0 = time.time()
        for XY in DtSpl:
            test_geom_nets(torch.from_numpy(XY)) # prints for all types of tests the -log_2(P) with P using 2st's
        print(f'testing 4 types of "likelihoods" for N=5 requires {int(1e03*(time.time()-t_0)/N)/1e03} secs / dataset @ n=1000')
        print('\n ----------- end anm type -----------\n')


def visualize_geom_net(XY):
    XY = XY.type(dtype); X,Y = XY[:,0].clone(), XY[:,1].clone()
    geom_net = GenerativeGeomNet(loss="sinkhorn", p=1)
    geom_net.set_data(X,Y)
    geom_net.plot_training_samples()

    print(f'C2ST test for XY = XY_hat')
    nnt = c2st.neural_net_c2st(geom_net._XY, geom_net._XY_hat.detach()) ; knnt = c2st.knn_c2st(geom_net._XY, geom_net._XY_hat.detach())
    print(f'c2st neural test: acc={nnt[0]}, P(T>acc)={nnt[1]},  (reject if pval < 1e-02)')
    print(f'c2st knn test: acc={knnt[0]}, P(T>acc)={knnt[1]},  (reject if pval < 1e-02)')

    print(f'parameter compression: {_score_wrapper(geom_net._net)}')


N, M = (100, 100) if not use_cuda else (1000, 1000) # if not same # pts for D1 D2
XY = anm_data(N, dim=2, SEED=22) #Y_j = data(N, dim=2)
# causal_fit()
# _sample_anms_geom_nets(N=5)
# visualize_geom_net(XY)
test_geom_nets(XY)
