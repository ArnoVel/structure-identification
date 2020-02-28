import torch
import numpy as np
import math

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# adapted from https://github.com/lopezpaz/classifier_tests/

def split_data(P,Q,split_perc=0.5):
    P = P if P.ndim == 2 else P.view(-1,1)
    Q = Q if Q.ndim == 2 else Q.view(-1,1)
    data = torch.cat([P,Q],0)
    labels = torch.cat([    torch.zeros(len(P)),
                            torch.ones(len(Q))], 0).view(-1,1)
    n = len(data) ; idx = torch.randperm(n)

    data, labels = data[idx] , labels[idx]
    split_idx = math.floor(n*split_perc)

    x_tr = data[:split_idx] ; x_te = data[split_idx:]
    y_tr = labels[:split_idx] ; y_te = labels[split_idx:]

    return x_tr, y_tr, x_te, y_te


def neural_net_c2st(P,Q, epochs=500, num_hiddens=20, return_test=False):
    x_tr, y_tr, x_te, y_te = split_data(P,Q)
    x_tr, y_tr, x_te, y_te = x_tr.type(dtype), y_tr.type(dtype),\
                             x_te.type(dtype), y_te.type(dtype)

    net = torch.nn.Sequential(  torch.nn.Linear(x_tr.shape[1],num_hiddens),
                                torch.nn.ReLU(),
                                torch.nn.Linear(num_hiddens,1),
                                torch.nn.Sigmoid()
                                )
    # basic ANN with ReLU activations and sigmoid, mapping R^D -> {0,1}
    loss = torch.nn.BCELoss()

    net = net.type(dtype)

    optim = torch.optim.Adam(net.parameters(), lr=1e-02)

    for i in range(epochs):
        optim.zero_grad()
        L = loss(net(x_tr), y_tr)
        L.backward()
        optim.step()
    net.eval()
    preds = (net(x_te) > 0.5).float()
    acc = ( preds == y_te).float().mean()
    # for len(P) >= 200 assume N(1/2, 1/4/n_te)
    cdf_val = torch.distributions.normal.Normal(0.5, math.sqrt(0.25/len(x_te))).cdf(acc)

    if return_test:
        # if true, we're looking to compute e.g. compression bounds
        test = {'acc':acc,
                'pval':1-cdf_val,
                'params':[p.detach() for p in net.parameters()]}
        return test
    else:
        return acc, 1 - cdf_val

def distances(X,Y):
    '''X,Y contains vector observations X_i, Y_j as rows'''
    d = -2 * X @ Y.t()
    d += (X * X).sum(dim=1, keepdims=True).expand_as(d)
    d += (Y * Y).sum(dim=1, keepdims=True).t().expand_as(d)
    return d

def knn_c2st(P,Q, k=None, return_test=False):
    x_tr, y_tr, x_te, y_te = split_data(P,Q)
    x_tr, y_tr, x_te, y_te = x_tr.type(dtype), y_tr.type(dtype),\
                             x_te.type(dtype), y_te.type(dtype)

    k = k if k is not None else math.sqrt(len(x_tr))
    t = math.ceil(k/2.0) ; k = math.ceil(k)
    pred_te = torch.zeros(y_te.shape).type(dtype)
    # distances(x_te,x_tr) gives as cols all the x_tr distances to x_te[i]
    d_ord, d_idx = torch.sort(distances(x_te,x_tr), dim=1)
    # d_idx[i] contains indexes for the closest x_tr's
    for i in range(len(x_te)):
        # only use top k closest to predict
        if (y_tr[d_idx[i][:k]].sum() > t):
            pred_te[i] = 1
    acc = (pred_te == y_te).float().mean()
    cdf_val = torch.distributions.normal.Normal(0.5, math.sqrt(0.25/len(x_te))).cdf(acc)
    if return_test:
        test = {'acc':acc,
                'pval':1-cdf_val}
        # what shoud we send to allow for decoding of y[1:n] | x[1:n] in the knn case?
        return test
    else:
        return acc, 1 - cdf_val

# to run quick tests in command line
# import torch ; from dependence import c2st ; P,Q = torch.randn(550), torch.randn(550)
# for mu in torch.linspace(0,10,20):
#      P,Q = torch.randn(550), torch.randn(550)+mu
#      print(c2st.knn_c2st(P,Q))
#      print(c2st.neural_net_c2st(P,Q))
#      print(f'end of loop for mu={mu}')
