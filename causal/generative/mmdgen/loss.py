import numpy as np
from scipy.stats import ttest_ind
import torch as th

from .Settings import SETTINGS


# courtesy of https://github.com/Diviyan-Kalainathan/

class TTestCriterion(object):
    """ A loop criterion based on t-test to check significance of results.
    Args:
        max_iter (int): Maximum number of iterations authorized
        runs_per_iter (int): Number of runs performed per iteration
        threshold (float): p-value threshold, under which the loop is stopped.
    """
    def __init__(self, max_iter, runs_per_iter, threshold=0.01):
        super(TTestCriterion, self).__init__()
        self.threshold = threshold
        self.max_iter = max_iter
        self.runs_per_iter = runs_per_iter
        self.iter = 0
        self.p_value = np.inf

    def loop(self, xy, yx):
        """ Tests the loop condition based on the new results and the
        parameters.
        Args:
            xy (list): list containing all the results for one set of samples
            yx (list): list containing all the results for the other set.
        Returns:
            bool: True if the loop has to continue, False otherwise.
        """
        if self.iter < 2:
            self.iter += self.runs_per_iter
            return True
        t_test, self.p_value = ttest_ind(xy, yx, equal_var=False)
        if self.p_value > self.threshold and self.iter < self.max_iter:
            self.iter += self.runs_per_iter
            return True
        else:
            return False


class MMDloss(th.nn.Module):

    def __init__(self, input_size, bandwidths=None):
        """Init the model."""
        super(MMDloss, self).__init__()
        if bandwidths is None:
            self.bandwidths = [0.01, 0.1, 1, 10, 100]
        else:
            self.bandwidths = [bandwidths]
        s = th.cat([th.ones([input_size, 1]) / input_size,
                    th.ones([input_size, 1]) / -input_size], 0)

        self.register_buffer('S', (s @ s.t()))

    def forward(self, x, y):

        X = th.cat([x, y], 0)

        # dot product between all combinations of rows in 'X'
        XX = X @ X.t()
        # dot product of rows with themselves
        X2 = (X * X).sum(dim=1).unsqueeze(0)

        ## exponent entries of the RBF kernel (without the sigma) for each
        ## combination of the rows in 'X'
        exponent = -2*XX + X2.expand_as(XX) + X2.t().expand_as(XX)
        #print('exponent shape',exponent.shape)
        val = [ self.S *(exponent * -bandwidth).exp()
                              for bandwidth in self.bandwidths]

        lossMMD = th.sum(sum(val))

        return lossMMD
