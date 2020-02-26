import numpy as np
import matplotlib.pyplot as plt
import time
from random import choices, choice, seed
from imageio import imread
from matplotlib import pyplot as plt
import torch
from geomloss import SamplesLoss
from functions.generators.generators import *
from functions.miscellanea import _write_nested, _plotter, GridDisplay
from dependence import c2st


use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

print(f'Using cuda? {"yes" if use_cuda else "no"}, data type is then {dtype}')

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

def fcm_gradient_flow(data, loss, lr=.05, loss_info=None, direction='->') :
    """updates net params along the gradient of the cost function, using a simple Euler scheme.

    Parameters:
        loss ((x_i,y_j) -> torch float number):
            Real-valued loss function.
        lr (float, default = .05):
            Learning rate, i.e. time step.
    """

    # Parameters for the gradient descent
    total_ = 5 ; num_displays = 6
    Nsteps = int(total_/lr)+1 # base was 5/lr
    display_its = [int(t/lr) for t in torch.linspace(0,total_,num_displays)]
    display = GridDisplay(num_items=num_displays, nrows=-1, ncols=3)

    print(f'display its: {display_its}')
    # Use colors to identify the particles
    colors = (10*data[:,0]).cos() * (10*data[:,1]).cos()
    colors = colors.detach().cpu().numpy()

    # Make sure that we won't modify the reference samples
    XY_, X_, Y_ = data.clone(), data[:,0:1].clone(), data[:,1:2].clone()
    XY_, X_, Y_ = XY_.type(dtype), X_.type(dtype), Y_.type(dtype)

    # get basic setup for FCM y = f(x,e ; params)
    num_hiddens = 20
    f_net = torch.nn.Sequential(torch.nn.Linear(2,num_hiddens),
                                torch.nn.ReLU(),
                                torch.nn.Linear(num_hiddens,num_hiddens),
                                torch.nn.ReLU(),
                                torch.nn.Linear(num_hiddens,1)
                                )
    f_net.register_buffer('noise',torch.Tensor(len(X_), 1))
    f_net = f_net.type(dtype)

    optim = torch.optim.Adam(f_net.parameters(), lr=lr)
    # We're going to perform gradient descent on Loss(α, β)
    # wrt. the positions x_i of the diracs masses that make up α:

    t_0 = time.time()
    for i in range(Nsteps):
        optim.zero_grad()
        # Compute cost and gradient
        f_net.noise.normal_()
        if direction == '->':
            Y_hat_ = f_net( torch.cat( [X_, f_net.noise], 1 ) )
            XY_hat_ = torch.cat( [X_, Y_hat_],1)
        elif direction == '<-':
            X_hat_ = f_net( torch.cat( [Y_, f_net.noise], 1 ) )
            XY_hat_ = torch.cat( [X_hat_, Y_] ,1)
        L_αβ = loss(XY_, XY_hat_)
        L_αβ.backward()
        if i in display_its : # display
            print(f'loop iter {i}')
            #print(f'check gradient and loss magnitudes: {g.norm().data, L_αβ.data}')
            def callback(ax, x_i, y_j, colors):
                plt.set_cmap("hsv")
                display_samples(ax, XY_, [(.55,.55,.95)])
                display_samples(ax, XY_hat_, colors)
                ax.set_title("t = {:1.2f}".format(lr*i))

                #plt.axis([0,1,0,1])
                #plt.gca().set_aspect('equal', adjustable='box')
                plt.xticks([], []); plt.yticks([], [])
                plt.tight_layout()
            display.add_plot(callback=(lambda ax: callback(ax, XY_, XY_hat_, colors)))


        # in-place modification of the tensor's values
        optim.step()

    display.fig.suptitle((f"Gradient flows with loss {loss_info};"+
                          f"\n T = {lr*i}, elapsed time: {int(1e03*(time.time() - t_0)/Nsteps)/1e03}s/it"+
                          f"\n on ANM data {c,m,bn}"),
                          fontsize=10)
    display.fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    print(f'C2ST test for XY = XY_hat')
    nnt = c2st.neural_net_c2st(XY_, XY_hat_.detach()) ; knnt = c2st.knn_c2st(XY_,XY_hat_.detach())
    print(f'c2st neural test: acc={nnt[0]}, P(T>acc)={nnt[1]},  (reject if pval < 1e-02)')
    print(f'c2st knn test: acc={knnt[0]}, P(T>acc)={knnt[1]},  (reject if pval < 1e-02)')

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

def causal_fit():
    N, M = (100, 100) if not use_cuda else (1000, 1000) # if not same # pts for D1 D2
    C = torch.rand(2,2) ; C = C.t() @ C
    XY = anm_data(N, dim=2, SEED=22) #Y_j = data(N, dim=2)
    loss_info = {"loss":"sinkhorn", "p":1, "blur":1e-01, "scaling":0.7}
    fcm_gradient_flow( data=XY,  loss=SamplesLoss(**loss_info) , lr=5e-02, loss_info=loss_info, direction='->')
    plt.show()
    fcm_gradient_flow( data=XY,  loss=SamplesLoss(**loss_info) , lr=5e-02, loss_info=loss_info, direction='<-')
    plt.show()

def c2st_tests(n=500, test='mu'):
    if test=='mu':
        for mu in torch.linspace(0,1,10):
            P,Q = torch.randn(550,2), torch.randn(550,2)+mu*torch.ones(1,2)
            print(c2st.knn_c2st(P,Q))
            print(c2st.neural_net_c2st(P,Q))
            print(f'end of loop for mu={mu}')


# causal_fit()
c2st_tests(n=500)
c2st_tests(n=1000)
