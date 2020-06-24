import numpy as np
import torch
from torch.nn import Module,Parameter
from torch.nn.functional import softmax, log_softmax
from pykeops.torch import Kernel, kernel_product
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import seaborn as sns

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# DISPLAY CONSTANTS



# main class

class GaussianMixture(Module):
    def __init__(self, M, sparsity=0, D=2):
        super(GaussianMixture, self).__init__()

        self.params = {'id': Kernel('gaussian(x,y)')}
        # We initialize our model with random blobs scattered across
        # the unit square, with a small-ish radius:
        self.mu = Parameter(torch.rand(M, D).type(dtype))
        self.A = 15 * torch.ones(M, 1, 1) * torch.eye(D, D).view(1, D, D)
        self.A = Parameter((self.A).type(dtype).contiguous())
        self.w = Parameter(torch.ones(M, 1).type(dtype))
        self.sparsity = sparsity
        self.D = D


    def update_covariances(self):
        """Computes the full covariance matrices from the model's parameters."""
        (M, D, _) = self.A.shape
        self.params['gamma'] = (torch.matmul(self.A, self.A.transpose(1, 2))).view(M, D * D) / 2


    def covariances_determinants(self):
        """Computes the determinants of the covariance matrices.

        N.B.: PyTorch still doesn't support batched determinants, so we have to
              implement this formula by hand.
        """
        S = self.params['gamma']
        if S.shape[1] == 2 * 2:
            dets = S[:, 0] * S[:, 3] - S[:, 1] * S[:, 2]
        elif S.shape[1 == 1*1]:
            dets = S[:, 0]
        else:
            raise NotImplementedError
        return dets.view(-1, 1)


    def weights(self):
        """Scalar factor in front of the exponential, in the density formula."""
        return softmax(self.w, 0) * self.covariances_determinants().sqrt()


    def weights_log(self):
        """Logarithm of the scalar factor, in front of the exponential."""
        return log_softmax(self.w, 0) + .5 * self.covariances_determinants().log()


    def likelihoods(self, sample):
        """Samples the density on a given point cloud."""
        self.update_covariances()
        #print([obj.shape for obj in [self.params['gamma'], sample, self.mu, self.weights()]])
        return kernel_product(self.params, sample, self.mu, self.weights(), mode='sum', backend="pytorch")


    def log_likelihoods(self, sample):
        """Log-density, sampled on a given point cloud."""
        self.update_covariances()
        #print([obj.shape for obj in [self.params['gamma'], sample, self.mu, self.weights_log()]])
        return kernel_product(self.params, sample, self.mu, self.weights_log(), mode='lse', backend="pytorch")


    def neglog_likelihood(self, sample):
        """Returns -log(likelihood(sample)) up to an additive factor."""
        ll = self.log_likelihoods(sample)
        #print(ll, [(np[0],np[1].shape) for np in self.named_parameters()])
        #print(self.params)
        log_likelihood = torch.mean(ll)
        # N.B.: We add a custom sparsity prior, which promotes empty clusters
        #       through a soft, concave penalization on the class weights.
        return -log_likelihood + self.sparsity * softmax(self.w, 0).sqrt().mean()


    def train(self, sample, num_iters=200, lr=1e-01):
        """ trains the model by maximizing the (penalized) log likelihood """
        self.loss = np.zeros(num_iters)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for iter in range(num_iters):
            self.optimizer.zero_grad()  # Reset the gradients (PyTorch syntax...).
            cost = self.neglog_likelihood(sample)  # Cost to minimize.
            #print(cost)
            cost.backward()  # Backpropagate to compute the gradient.
            self.optimizer.step()

            self.loss[iter] = cost.data.cpu().numpy()

    def plot(self, sample, ax=None):
        """Displays the model."""
        if ax == None:
            ax = plt
        if self.D==1:
            x = sample.data.cpu().numpy()
            mu, sigma = x.mean(), x.std()
            # for 1D plots
            line = torch.linspace(x.min(),x.max(),500).contiguous().type(dtype).view(-1,1)
            if ax == plt:
                sns.rugplot(x,color='k', lw=0.1, alpha=0.05, height=0.02)
            else:
                sns.rugplot(x,color='k', lw=0.1, alpha=0.05, height=0.02, ax=ax)
            heatmap = self.likelihoods(line)
            heatmap = heatmap.view(res, 1).data.cpu().numpy().ravel()
            # reshape as a "background" density and somewhat normalize
            line_cpu = line.data.cpu().numpy().ravel()
            # integral approx with uniform grid on [0,1], delta/N spacing
            spacing = line_cpu.max() - line_cpu.min()
            heatmap /= (heatmap.mean()*spacing)
            ax.fill_between(line_cpu,heatmap,0,color='r',alpha=0.4, label="GMM Estimate")
        if self.D==2:
            # Create a uniform grid on the unit square:
            res = 1000
            xlow,ylow,xhi,yhi = sample[:,0].min() - sample[:,0].std(),\
                                sample[:,1].min() - sample[:,1].std(),\
                                sample[:,0].max() + sample[:,0].std(),\
                                sample[:,1].max() + sample[:,1].std()

            xlow,ylow,xhi,yhi = xlow.cpu().numpy(),ylow.cpu().numpy(),xhi.cpu().numpy(),yhi.cpu().numpy()
            xticks,yticks = np.linspace(xlow,xhi, res + 1)[:-1],\
                            np.linspace(ylow,yhi, res + 1)[:-1]
            # for 2D plots
            X, Y = np.meshgrid(xticks, yticks)
            grid = torch.from_numpy(np.vstack((X.ravel(), Y.ravel())).T).contiguous().type(dtype)
            # Heatmap:
            #plt.gcf()
            heatmap = self.likelihoods(grid)
            heatmap = heatmap.view(res, res).data.cpu().numpy()  # reshape as a "background" image
            idx = np.argsort(heatmap.ravel())

            scale = np.amax(np.abs(heatmap[:]))
            ax.imshow(-heatmap, interpolation='bilinear', origin='lower',
                       vmin=-scale, vmax=scale, cmap=cm.RdBu,
                       extent=(xlow,xhi,ylow,yhi))

            # Log-contours:
            log_heatmap = self.log_likelihoods(grid)
            log_heatmap = log_heatmap.view(res, res).data.cpu().numpy()

            scale = np.amax(np.abs(log_heatmap[:]))
            levels = np.linspace(-scale, scale, 41)

            ax.contour(log_heatmap, origin='lower', linewidths=1., colors="#C8A1A1",
                        levels=levels, extent=(xlow,xhi,ylow,yhi))

            # Scatter plot of the dataset:
            xy = sample.data.cpu().numpy()
            ax.scatter(xy[:, 0], xy[:, 1], 100 / len(xy), color='k')
            plt.axis([xlow,xhi,ylow,yhi])
