# Fitting Gaussian Mixture with variable number of classes

Here we adapt and follow the [code from PyKeOps](https://www.kernel-operations.io/keops/_auto_tutorials/gaussian_mixture/plot_gaussian_mixture.html#sphx-glr-auto-tutorials-gaussian-mixture-plot-gaussian-mixture-py)

## Two dimensional case

### Base example

This corresponds to the initial example. We obtain the following, as expected

![](./dim-two/gmm_fit_iter499.png?raw=true)

### ANM example

What if we attempt to fit an ANM without a causal direction,
only using its joint distribution?
