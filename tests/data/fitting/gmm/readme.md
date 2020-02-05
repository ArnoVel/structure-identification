# Fitting Gaussian Mixture with variable number of classes

Here we adapt and follow the [code from PyKeOps](https://www.kernel-operations.io/keops/_auto_tutorials/gaussian_mixture/plot_gaussian_mixture.html#sphx-glr-auto-tutorials-gaussian-mixture-plot-gaussian-mixture-py)

## Two dimensional case
The initial example would fit 2d mixtures to jointly distributed data.

### Base example

This corresponds to the initial example. We obtain the following, as expected

![](./dim-two/base_ex_499its.png?raw=true)

### ANM example

What if we attempt to fit an ANM without a causal direction,
only using its joint distribution?

**Mixture of 2 Uniforms as cause, spline mechanism:**  
![](./dim-two/anm_ex_unif_spline_499its.png?raw=true)

**Mixture of 2 Triangles as cause, spline mechanism:**
![](./dim-two/anm_ex_tri_spline_499its.png?raw=true)

**Mixture of 2 Gaussians as cause, spline mechanism:**
![](./dim-two/anm_ex_gauss_spline_499its.png?raw=true)


## One dimensional case

We adapt the code in order to fit 1D mixtures.
Our goal is to compute a reasonable approximation of the complexity of a distribution,
using the number of mixture components of a 'best' GMM approximation.
This either corresponds to the number of modes of the distribution, or roughly
to the complexity of the shape of the density.


### Two gaussians
