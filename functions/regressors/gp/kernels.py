import torch


def tensorize(scalar, device=None):
    if device is not None:
        return torch.Tensor([scalar]).to(device)
    else:
        return torch.Tensor([scalar])


class Kernel(torch.nn.Module):

    """Base kernel."""

    def __add__(self, other):
        """Sums two kernels together.
        Args:
            other (Kernel): Other kernel.
        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.add)

    def __mul__(self, other):
        """Multiplies two kernels together.
        Args:
            other (Kernel): Other kernel.
        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.mul)

    def __sub__(self, other):
        """Subtracts two kernels from each other.
        Args:
            other (Kernel): Other kernel.
        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.sub)

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        raise NotImplementedError


class AggregateKernel(Kernel):

    """An aggregate kernel."""

    def __init__(self, first, second, op):
        """Constructs an AggregateKernel.
        Args:
            first (Kernel): First kernel.
            second (Kernel): Second kernel.
            op (Function): Operation to apply.
        """
        super(Kernel, self).__init__()
        self.first = first
        self.second = second
        self.op = op

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        first = self.first(xi, xj, *args, **kwargs)
        second = self.second(xi, xj, *args, **kwargs)
        return self.op(first, second)


class RBFKernel(Kernel):

    """Radial-Basis Function Kernel."""

    def __init__(self, length_scale=None, sigma_s=None, eps=1e-6,
                        device=None):
        """Constructs an RBFKernel.
        Args:
            length_scale (Tensor): Length scale.
            sigma_s (Tensor): Signal standard deviation.
            eps (float): Minimum bound for parameters.
        """
        super(Kernel, self).__init__()
        init_ls, init_sig = torch.randn(1) if length_scale is None else length_scale,\
                            torch.randn(1) if sigma_s is None else sigma_s
        self.length_scale = torch.nn.Parameter(init_ls)
        self.sigma_s = torch.nn.Parameter(init_sig)
        self._eps = eps
        self._device = device
        if self._device is not None:
            self.cuda()

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        length_scale = (self.length_scale**-2).clamp(self._eps, 1e5)
        var_s = (self.sigma_s**2).clamp(self._eps, 1e5)
        if self._device is not None:
            M = torch.eye(xi.shape[1]).to(self._device) * length_scale
        else:
            M = torch.eye(xi.shape[1]) * length_scale
        dist = mahalanobis_squared(xi, xj, M)
        return var_s * (-0.5 * dist).exp()

class MaternKernel(Kernel):

    """Radial-Basis Function Kernel."""

    def __init__(self, length_scale=None, sigma_s=None, eps=1e-6,
                        device=None):
        """Constructs an RBFKernel.
        Args:
            length_scale (Tensor): Length scale.
            sigma_s (Tensor): Signal standard deviation.
            eps (float): Minimum bound for parameters.
        """
        super(Kernel, self).__init__()
        init_ls, init_sig = torch.randn(1).abs() if length_scale is None else length_scale,\
                            torch.randn(1).abs() if sigma_s is None else sigma_s
        self.length_scale = torch.nn.Parameter(init_ls)
        self.sigma_s = torch.nn.Parameter(init_sig)
        self._eps = eps
        self._device = device
        if self._device is not None:
            self.cuda()

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        length_scale = (self.length_scale**-2).clamp(self._eps, 1e5)
        var_s = (self.sigma_s**2).clamp(self._eps, 1e5)
        if self._device is not None:
            M = torch.eye(xi.shape[1]).to(self._device) * length_scale
        else:
            M = torch.eye(xi.shape[1]) * length_scale
        dist = mahalanobis_squared(xi, xj, M)
        dist = dist.sqrt()
        K = dist * tensorize(5.0, device=self._device).sqrt()
        K = (tensorize(1.0, device=self._device) + K +
             K.pow(2) / tensorize(3.0, device=self._device)) * (-K).exp()
        return K


class WhiteNoiseKernel(Kernel):

    """White noise kernel."""

    def __init__(self, sigma_n=None, eps=1e-6, max_noise=1e5 ,device=None):
        """Constructs a WhiteNoiseKernel.
        Args:
            sigma_n (Tensor): Noise standard deviation.
            eps (float): Minimum bound for parameters.
        """
        super(Kernel, self).__init__()
        init = torch.randn(1).abs() if sigma_n is None else sigma_n
        self.sigma_n = torch.nn.Parameter(init)
        self._eps = eps
        self._max_noise = max_noise
        self._device = device
    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        var_n = (self.sigma_n**2).clamp(self._eps, self._max_noise)
        if self._device is not None:
            var_n = var_n.to(self._device)
        return var_n

def mahalanobis_squared(xi, xj, VI=None):
    """Computes the pair-wise squared mahalanobis distance matrix as:
        (xi - xj)^T V^-1 (xi - xj)
    Args:
        xi (Tensor): xi input matrix.
        xj (Tensor): xj input matrix.
        VI (Tensor): The inverse of the covariance matrix, default: identity
            matrix.
    Returns:
        Weighted matrix of all pair-wise distances (Tensor).
    """
    xi, xj = xi.double(), xj.double()
    if VI is None:
        xi_VI = xi
        xj_VI = xj
    else:
        VI = VI.double()
        xi_VI = xi.mm(VI)
        xj_VI = xj.mm(VI)

    D = (xi_VI * xi).sum(dim=-1).reshape(-1, 1) \
      + (xj_VI * xj).sum(dim=-1).reshape(1, -1) \
      - 2 * xi_VI.mm(xj.t())
    return D
