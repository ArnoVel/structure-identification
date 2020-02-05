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

**Mixture of Uniforms as cause, spline mechanism:**  
![](./dim-two/anm_ex_mixtunif_spline.png?raw=true)

**Mixture of Gaussians as cause, spline mechanism:**
![](./dim-two/anm_ex_gmm_spline499.png?raw=true)

**Uniform as cause, tanhSum mechanism:**
![](./dim-two/anm_ex_uniform_tanhsum499.png?raw=true)


## One dimensional case

We adapt the code in order to fit 1D mixtures.
Our goal is to compute a reasonable approximation of the complexity of a distribution,
using the number of mixture components of a 'best' GMM approximation.
This either corresponds to the number of modes of the distribution, or roughly
to the complexity of the shape of the density.


### Two Triangles
![](./dim-one/triangles/sparsity_1/gmm_fit_iter499.png)

### Two Uniforms
![](./dim-one/unifs/overlapping/sparsity_1/seed_0/gmm_fit_iter499.png)