# Types of Cause Effect Pairs
We work with bivariate random variables $(X,Y) \sim P_{X,Y}$ generated as
$$ Y = f(X) + N,\qquad N\sim P_N,\; X\sim P_X$$
In the above, we call
* $X$ **cause** variable,
* $Y$ the **effect** variable.
* $N$ the **noise**, which is any centered independent $\mathbb{R}$-R.V.
* $f$ the (causal) **mechanism** which maps $X$ to $Y$.

## Cause Variables
### Uniform
![](./cause/uniform.png?raw=true)

### Uniform Mixture
![](./cause/uniform_mixture.png?raw=true)

### Gaussian Mixture
![](./cause/gaussian_mixture.png?raw=true)

### SubGaussian Mixture
![](./cause/subgaussian_mixture.png?raw=true)

### SuperGaussian Mixture
![](./cause/supergaussian_mixture.png?raw=true)

### Sub & Super Gaussian Mixture
![](./cause/supergaussian_mixture.png?raw=true)

## Mechanisms
We sample random mechanisms $f\sim P_F$, where $F$ corresponds to one of the following families
### Cubic Splines
![](./mechanism/cubic_spline.png?raw=true)

### Matern 2.5 Shift/Scale/Amplitude Sums
This name refers to a function of the form
$$ f(x) = \sum_{i}^K a_i\cdot f_i(\mu_i + \sigma_i\cdot x) $$
with all the parameters $\{K\}\cup\{a_i,\mu_i,\sigma_i,\lambda_i\}_i$ are randomized (and $\lambda_i$ is the matern bandwidth)
![](./mechanism/matern_sums.png?raw=true)

### Sigmoid AM
![](./mechanism/sigmoid_am.png?raw=true)

### Tanh Shift/Scale/Amplitude Sums
This name refers to a function of the form
$$ f(x) = \sum_{i}^K a_i\cdot f_i(\b_i\cdot(x+c_i)) $$
with all the parameters $\{K\}\cup\{a_i,\b_i,\b_i,\lambda_i\}_i$ are randomized
![](./mechanism/tanh_sum.png?raw=true)
