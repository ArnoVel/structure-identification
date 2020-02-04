## List about hyperparameters and resulting GaussianMixture

### Powers of gaussians

* M=30,sparsity=20, seed= 0
  --> converges to bimodal, as an approximation of 3 modes..
  --> might be 'too sparse'
  --> worse for seed=1996,911, converges to 1 mode and MLE Gaussian
  --> usually converges after 150/200 its

* sparsity=10, seed= 0
  same. Assume rest is same

* sparsity=1, seed = 0
  -> converges to very good approximation to true dist
  -> 8 weights bigger than 1e-5
  -> 4 weights bigger than 1e-4
  -> consider 4 modes, 4th might come from skew/kurtosis
  -> when seed=16,12,1996,911
  get 3 modes balanced, 3 balanced, 3 balanced,

### Uniforms
#### separated
* sparsity=1, seed = 0
  -> converges to two packs of gaussians, one for each uniform
  -> 8 weights bigger than 1e-4, 4 gaussians per uniform
  -> when seed=16,12,1996,911 similar, oscillates 7/8 gaussians total
  -> how to combine estimated parameters to reflect 2 modes?

* sparsity=10
  -> goes 'correctly' to bimodal

#### overlapping

* sparsity=1, seed = 0
  --> converges to healthy pack of 6 gaussians
  seed 12: pack of 5, quite similar but changes "a lot"
