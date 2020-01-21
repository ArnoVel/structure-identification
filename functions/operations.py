import torch

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
