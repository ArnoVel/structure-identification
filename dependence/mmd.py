import torch
from scipy.stats import gamma
from functions.kernels import RBF, SumIdentical, SumKernels
from functions.operations import block_matrix, cross_average, unblock_matrix

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

def numpy(tensor):
    return tensor.cpu().numpy()

class MMD(torch.nn.Module):
    def __init__(self,n_x, n_y, kernels=None, params=None,
                      unbiased=False, device=DEVICE):
        super(MMD, self).__init__()
        if not isinstance(kernels,list):
            # assume unique element
            kernels = [kernels]

        if kernels is None:
            params = [{'bandwidth':10**(i)} for i in range(-2,2)]
            kernels = [RBF for p in params]

        elif params is None:
            params = [{'bandwidth':10**(i)} for i in range(-2,2)]
            kernels = [kernels[0] for p in params]

        kerlist = [ker(**p, precompute=True) for (ker,p) in zip(kernels,params)]
        self.KernelFunc = SumKernels(kernel_list=kerlist)

        S = cross_average(n_x,n_y,unbiased)
        self.register_buffer('S', S)

        self.n_x, self.n_y = n_x, n_y
        self.unbiased = unbiased

    def forward(self,X,Y):
        # think of M@M.t() as being block-wise [XX, XY; YX, YY]
        M = torch.cat([X, Y], 0)
        self.K = self.KernelFunc(M)
        # use S to incorporate the MMD factors
        mmd_vals = self.S * self.K
        self.statistic = mmd_vals.sum()
        return self.statistic

    def GammaProb(self,X,Y, alpha=0.05):
        ## test only defined for n_x = n_y,
        # therefore drop the sample that's too large
        if self.n_x != self.n_y or not hasattr(self,'K'):
            n = min(self.n_x,self.n_y)
            X, Y = X[:n,:], Y[:n,:]
            M = torch.cat([X, Y], 0)
            self.K = self.KernelFunc(M)
        # use S to incorporate the MMD factors
        Kxx, Kxy, Kyx, Kyy = unblock_matrix(n,n,self.K)
        H = Kxx + Kyy - Kxy - Kyx
        statistic = H.sum() /n/n
        # but we use the distribution of n*MMD^2
        statistic = n*statistic
        # under H0, what is the mean of the biased statistic?
        # using the formulation of 2-variable expectations of V-stats vs. U-stats
        # if V=h(X,Y) is V-stat for T, E(V) = (1-1/n)*E(U) + E(h(X,X))
        # in our case, X and Y are z_i's, z_i=(x_i,y_i) , under H0 E(U)= MMD^2(P,P)=0
        # the only thing left is E(h(Z,Z)) = E(k(X,X)+k(Y,Y) - 2*K(X,Y))
        # in practice X ~ P , Y ~ Q = P ==> E(k(X,Y)) = E(k(X,X)) = E(k(Y,Y))
        # but in this case it is safer to estimate quantities from data.
        # using empirical means, as k(x_i,x_i) = k(y_i,y_i) = CONST , we get 2*1 - 2*E(k(X,Y))
        const = self.KernelFunc(torch.zeros(1,1).to(DEVICE))
        mean = 2/n * (const - Kxy.diagonal().sum()/n)
        # var formula is 2/(n*(n-1)) times expectation of h^2(z,z')
        # do as the original paper and estimate variance unbiasedly
        H = H.fill_diagonal_(0)
        var = (2/(n*(n-1)))* (H*H).sum()/(n*(n-1))
        # gamma distribution for n*MMD^2
        # use scipy as no icdf gamma in torch
        self.alpha, self.beta = numpy(mean**2/var), numpy(n*var/mean)
        #print(f'mean: {mean}, var: {var}, alpha:{self.alpha}, beta:{self.beta}')
        self.test_cdf = (lambda x: 1.0 - gamma.cdf(x, a=self.alpha, loc=0, scale=self.beta))
        self.test_thresh = gamma.ppf(1-alpha, a=self.alpha, loc=0, scale=self.beta)
        self.gamma_test_stat = statistic # not the same as stat, it is n*MMD^2 biased
        return self.gamma_test_stat, self.test_thresh
