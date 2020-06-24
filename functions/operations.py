import torch

def numpify(tensor):
    if isinstance(tensor,torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor

def block_matrix(A,B,C,D):
    ''' returns [A,B; C,D] if dims match'''
    mat = torch.cat([
                torch.cat([A,B],1),
                torch.cat([C,D],1)
                ],0)
    return mat

def unblock_matrix(n,m,M):
    ''' assumes [A,B;C,D], with dims
        [n,n] , [n,m] , [m,n], [m,m]'''
    return M[:n,:n], M[:n,n:], M[n:,:n], M[n:,n:]

def cross_average(n_x,n_y, unbiased=False):
    if unbiased:
        # following the paper from Gretton
        # might be negative, refer to
        # https://arxiv.org/pdf/0805.2368.pdf (eqn 13)
        A,B,C,D =  torch.ones([n_x,n_x]).fill_diagonal_(0)/(n_x*(n_x-1)),\
                  -torch.ones([n_x,n_y]).fill_diagonal_(0)/(n_x*n_y),\
                  -torch.ones([n_y,n_x]).fill_diagonal_(0)/(n_x*n_y),\
                   torch.ones([n_y,n_y]).fill_diagonal_(0)/(n_y*(n_y-1))

        S = block_matrix(A,B,C,D)
    else:
        S = torch.cat([torch.ones([n_x, 1]) / n_x,
                    torch.ones([n_y, 1]) / -n_y], 0)
        S = S @ S.t()
    return S

def _smooth_ecdf(vals, eps=5e-03, compress=True, sort=False):
    if vals.ndim > 1:
        if vals.shape[0]!=1 and vals.shape[1]!=1:
            raise ValueError("Only accepts 1d data",vals.shape)
        else:
            vals = vals.flatten()


    idx = torch.argsort(vals, dim=0)
    v = vals[idx]
    n = v.nelement()
    diffs = v.view(-1,1) - v.view(-1,1).t()
    # for an entry (i,j) gives X_i - X_j
    # we want to sum_j ind{X_j < X_i}
    diffs = (diffs/eps).sigmoid()
    if compress:
        normalizer =  n+1
    else:
        normalizer = n
    cdvals = diffs.sum(dim=1) / normalizer
    if sort:
        return v.flatten(),cdvals.flatten()
    else:
        # invert the permutation
        idx_inv = torch.zeros(idx.nelement())
        for i in range(idx.nelement()):
            idx_inv[idx[i]] = i
        idx_inv = torch.Tensor(idx_inv).long()
        return v.flatten()[idx_inv],cdvals.flatten()[idx_inv]

def _mvar_smooth_ecdf(vals, eps=5e-03, compress=True, sort=False, x_vals=True):
    ''' supposes [N,D] tensor with N the number of points, and each D-dimensional'''
    # for now use python loops, but some mapreduce might be better
    # returns a [2,N,D] tensor
    rets = []
    for d in range(vals.shape[1]):
        # the stack on flat tensors op returns a [nelem,vecsize], with nelem=len(list)
        rets.append(torch.stack(_smooth_ecdf(vals=vals[:,d], eps=eps, compress=compress, sort=sort)))
    # on k-order tensors will same shapes, just creates a new din (the 0 dim)
    # and stacks all the tensors along the dim, getting [nelem,*[item shape]]
    # so here [D,2,N]
    rets = torch.stack(rets) ; rets = torch.transpose(torch.transpose(rets,0,1),1,2)
    if x_vals:
        return rets
    else:
        # only return cdf values
        return rets[1,:,:]

def _smooth_max(vals,eps=5e-03):
    if vals.ndim > 1:
        if vals.shape[0]!=1 and vals.shape[1]!=1:
            raise ValueError("Only accepts 1d data",vals.shape)
        else:
            vals = vals.flatten()
    # if eps < 1e-01 , generally the approximation is quite good
    return (vals/eps).logsumexp(0)*eps

def _smooth_min(vals,eps=5e-03):
    if vals.ndim > 1:
        if vals.shape[0]!=1 and vals.shape[1]!=1:
            raise ValueError("Only accepts 1d data",vals.shape)
        else:
            vals = vals.flatten()
    # if eps < 1e-01 , generally the approximation is quite good
    return -((-vals/eps).logsumexp(0)*eps)
