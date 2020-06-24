"""Gaussian Process models."""

import torch
import warnings
import numpy as np
from dependence.hsic import HSIC


class IGaussianProcess(torch.nn.Module):

    """Base Gaussian Process regressor."""

    def __init__(self):
        """Constructs an IGaussianProcess."""
        super(IGaussianProcess, self).__init__()
        self._is_set = False

    @property
    def is_set(self):
        """Whether the training data is set or not."""
        return self._is_set

    def set_data(self, X, Y, normalize_y=True, reg=1e-5):
        """Set the training data.
        Args:
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
            normalize_y (bool): Whether to normalize the outputs.
        """
        raise NotImplementedError

    def loss(self):
        """Computes the loss as the negative marginal log likelihood."""
        raise NotImplementedError

    def forward(self,
                x,
                return_mean=True,
                return_var=False,
                return_covar=False,
                return_std=False,
                **kwargs):
        """Computes the GP estimate.
        Args:
            x (Tensor): Inputs.
            return_mean (bool): Whether to return the mean.
            return_covar (bool): Whether to return the full covariance matrix.
            return_var (bool): Whether to return the variance.
            return_std (bool): Whether to return the standard deviation.
        Returns:
            Tensor or tuple of Tensors.
            The order of the tuple if all outputs are requested is:
                (mean, covariance, variance, standard deviation).
        """
        raise NotImplementedError

    def fit(self, tol=1e-6, reg_factor=10.0, max_reg=1.0, max_iter=1000):
        """Fits the model.
        Args:
            tol (float): Tolerance.
            reg_factor (float): Regularization multiplicative factor.
            max_reg (float): Maximum regularization term.
            max_iter (int): Maximum number of iterations.
        Returns:
            Number of iterations.
        """
        raise NotImplementedError


class GaussianProcess(IGaussianProcess):

    """Gaussian Process regressor.
    This is meant for multi-input, single-output functions.
    It still works for multi-output functions, but will share the same
    hyperparameters for each output. If that is not wanted, use
    `MultiGaussianProcess` instead.
    """

    def __init__(self, kernel, sigma_n=None, eps=1e-4,
                       losstype='nll',device=None):
        """Constructs a GaussianProcess.
        Args:
            kernel (Kernel): Kernel.
            sigma_n (Tensor): Noise standard deviation.
            eps (float): Minimum bound for parameters.
        """
        super(GaussianProcess, self).__init__()
        self.kernel = kernel
        self.sigma_n = torch.nn.Parameter(
            torch.randn(1) if sigma_n is None else sigma_n)
        self._eps = eps
        self._is_set = False
        self._device = device
        if self._device is not None:
            self.to(self._device)
        self.loss_type = losstype

    def _update_k(self):
        """Updates the K matrix."""
        X = self._X
        Y = self._Y

        # Compute K and guarantee it's positive definite.
        var_n = (self.sigma_n**2).clamp(self._eps, 1e5)
        K = self.kernel(X, X).double()
        K = (K + K.t()).mul(0.5)
        if self._device is not None:
            self._K = K + (self._reg + var_n) * torch.eye(X.shape[0]).to(self._device)
        else:
            self._K = K + (self._reg + var_n) * torch.eye(X.shape[0])

        # Compute K's inverse and Cholesky factorization.
        # We can't use potri() to compute the inverse since it's derivative
        # isn't implemented yet.
        self._L = torch.cholesky(self._K)
        self._K_inv = self._K.inverse()

    def set_data(self, X, Y, normalize_y=True, reg=1e-5, hsic_unbiased=False):
        """Set the training data.
        Args:
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
            normalize_y (bool): Whether to normalize the outputs.
        """
        self._non_normalized_Y = Y.double()

        if normalize_y:
            Y_mean = torch.mean(Y, dim=0)
            Y_variance = torch.std(Y, dim=0)
            Y = (Y - Y_mean) / Y_variance

        if self.loss_type == 'hsic' or self.loss_type == 'hsic-gamma':
                self._hsic = HSIC(n=X.shape[0],unbiased=hsic_unbiased)

        self._X = X.double()
        self._Y = Y.double()
        self._reg = reg
        self._update_k()
        self._is_set = True

    def loss(self, nll_factor=1.0):
        """Computes the loss as the negative marginal log likelihood."""

        if not self._is_set:
            raise RuntimeError("You must call set_data() first")

        Y = self._Y
        self._update_k()
        K_inv = self._K_inv.double()

        # Compute the log likelihood.
        log_likelihood_dims = -0.5 * Y.t().mm(K_inv.mm(Y)).sum(dim=0)
        log_likelihood_dims -= self._L.diag().log().sum()
        log_likelihood_dims -= self._L.shape[0] / 2.0 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(dim=-1)

        if self.loss_type=="nll":
            return -log_likelihood
        elif self.loss_type=="hsic":
            # careful, as X is normalized, renormalize the predictions and residuals
            _F = self(self._X) ; _F = (_F - _F.mean(0)) / _F.std(0)
            _residuals = self._Y - _F ; _residuals = (_F - _residuals.mean(0)) / _residuals.std(0)
            indep_criterion = self._hsic(self._X, _residuals)
            return indep_criterion - nll_factor * log_likelihood

        elif self.loss_type=="hsic-gamma":
            # careful, as X is normalized, renormalize the predictions and residuals
            _F = self(self._X) ; _F = (_F - _F.mean(0)) / _F.std(0)
            _residuals = self._Y - _F ; _residuals = (_F - _residuals.mean(0)) / _residuals.std(0)
            indep_criterion = self._hsic(self._X, _residuals)
            # return n*MMD , which is used to perform gamma test
            return self._X.shape[0]* indep_criterion - nll_factor * log_likelihood

        else:
            raise NotImplementedError("This loss type isn't yet implemented",self.loss_type)

    def forward(self,
                x,
                return_mean=True,
                return_var=False,
                return_covar=False,
                return_std=False,
                **kwargs):
        """Computes the GP estimate.
        Args:
            x (Tensor): Inputs.
            return_mean (bool): Whether to return the mean.
            return_covar (bool): Whether to return the full covariance matrix.
            return_var (bool): Whether to return the variance.
            return_std (bool): Whether to return the standard deviation.
        Returns:
            Tensor or tuple of Tensors.
            The order of the tuple if all outputs are requested is:
                (mean, covariance, variance, standard deviation).
        """
        if not self._is_set:
            raise RuntimeError("You must call set_data() first")

        X = self._X
        Y = self._Y
        K_inv = self._K_inv

        # Kernel functions.
        K_ss = self.kernel(x, x)
        K_s = self.kernel(x, X)

        # Compute mean.
        outputs = []
        if return_mean:
            # Non-normalized for scale.
            mean = K_s.mm(K_inv.mm(self._non_normalized_Y))
            outputs.append(mean)

        # Compute covariance/variance/standard deviation.
        if return_covar or return_var or return_std:
            covar = K_ss - K_s.mm(K_inv.mm(K_s.t()))
            if return_covar:
                outputs.append(covar)
            if return_var or return_std:
                var = covar.diag().reshape(-1, 1)
                if return_var:
                    outputs.append(var)
                if return_std:
                    std = var.sqrt()
                    outputs.append(std)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def fit(self, tol=1e-6, reg_factor=10.0, lr=1e-02,
                  max_reg=1.0, max_iter=1000, nll_factor=1.0):
        """Fits the model.
        Args:
            tol (float): Tolerance.
            reg_factor (float): Regularization multiplicative factor.
            max_reg (float): Maximum regularization term.
            max_iter (int): Maximum number of iterations.
            nll_factor (float): if hsic, how fix lagrange multiplier for nll
        Returns:
            Number of iterations.
        """
        if not self._is_set:
            raise RuntimeError("You must call set_data() first")

        opt = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr, weight_decay=1e-04)

        while self._reg <= max_reg:
            try:
                curr_loss = np.inf
                n_iter = 0

                while n_iter < max_iter:
                    opt.zero_grad()

                    prev_loss = self.loss(nll_factor=nll_factor)
                    prev_loss.backward(retain_graph=True)
                    opt.step()

                    curr_loss = self.loss()
                    dloss = curr_loss - prev_loss
                    n_iter += 1
                    # print(dloss) ; print(dloss / curr_loss)
                    if (dloss/curr_loss).abs() <= tol:
                        break

                return n_iter

            except RuntimeError as rt_err:
                # Increase regularization term until it succeeds.

                if 'cholesky' in str(rt_err):
                    self._reg *= reg_factor
                    continue
                else:
                    print('no singular K, raising error')
                    raise RuntimeError


        warnings.warn("exceeded maximum regularization: did not converge")

class MultiGaussianProcess(IGaussianProcess):

    """
    Layer of abstraction to estimate vector-valued functions with a separate
    Gaussian Process for each output dimension.
    This learns separate hyperparameters for each output dimension instead of
    a single set for all dimensions.
    """

    def __init__(self, kernels, *args, **kwargs):
        """Constructs a MultiGaussianProcess.
        Args:
            kernels (list<Kernel>): List of kernels to use for each dimension.
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
            *args, **kwargs: Additional positional and key-word arguments to
                each Gaussian Process constructor.
        """
        super(MultiGaussianProcess, self).__init__()
        [
            self.add_module("process_{}".format(i),
                            GaussianProcess(kernel, *args, **kwargs))
            for i, kernel in enumerate(kernels)
        ]
        self._processes = [
            getattr(self, "process_{}".format(i)) for i in range(len(kernels))
        ]

    def set_data(self, X, Y, normalize_y=True):
        """Set the training data.
        Args:
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
            normalize_y (bool): Whether to normalize the outputs.
        """
        self._X = X
        self._Y = Y
        for i, gp in enumerate(self._processes):
            gp.set_data(X, Y[:, i].reshape(-1, 1), normalize_y)
        self._is_set = True

    def loss(self):
        """Computes the loss as the negative marginal log likelihood."""
        if not self._is_set:
            raise RuntimeError("You must call set_data() first")

        loss = torch.tensor(0.0)
        for gp in self._processes:
            loss += gp.loss()
        return loss

    def fit(self, *args, **kwargs):
        """Fits the model.
        Args:
            *args, **kwargs: Additional positional and key-word arguments to
                pass to each gaussian process's fit() function.
        Returns:
            Total number of iterations.
        """
        if not self._is_set:
            raise RuntimeError("You must call set_data() first")

        iters = 0
        for gp in self._processes:
            iters += gp.fit(*args, **kwargs)
        return iters

    def forward(self, x, *args, **kwargs):
        """Computes the GP estimate.
        Args:
            x (Tensor): Inputs.
            *args, **kwargs: Additional positional and key-word arguments to
                pass to each gaussian process's forward() function.
        Returns:
            Tensor or tuple of Tensors.
            The order of the tuple if all outputs are requested is:
                (mean, covariance, variance, standard deviation).
        """
        if not self._is_set:
            raise RuntimeError("You must call set_data() first")

        outputs = np.array([gp(x, *args, **kwargs) for gp in self._processes])

        if outputs.ndim > 1:
            outputs = [
                torch.cat(tuple(outputs[:, i]), dim=-1)
                for i in range(outputs.shape[1])
            ]
        else:
            outputs = torch.cat(tuple(outputs), dim=-1)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)
