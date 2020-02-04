# Types of Cause Effect Pairs
We work with bivariate random variables ![(X,Y) \sim P_{X,Y}](https://render.githubusercontent.com/render/math?math=(X%2CY)%20%5Csim%20P_%7BX%2CY%7D) generated as  

![Y = f(X) + N,\qquad N\sim P_N,\; X\sim P_X](https://render.githubusercontent.com/render/math?math=Y%20%3D%20f(X)%20%2B%20N%2C%5Cqquad%20N%5Csim%20P_N%2C%5C%3B%20X%5Csim%20P_X)  

In the above, we call
* ![$X$](https://render.githubusercontent.com/render/math?math=%24X%24) **cause** variable,
* ![$Y$](https://render.githubusercontent.com/render/math?math=%24Y%24) the **effect** variable.
* ![$N$](https://render.githubusercontent.com/render/math?math=%24N%24) the **noise**, which is any centered independent ![$\mathbb{R}$](https://render.githubusercontent.com/render/math?math=%24%5Cmathbb%7BR%7D%24)-R.V.
* ![$f$](https://render.githubusercontent.com/render/math?math=%24f%24) the (causal) **mechanism** which maps ![$X$](https://render.githubusercontent.com/render/math?math=%24X%24) to ![$Y$](https://render.githubusercontent.com/render/math?math=%24Y%24).

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
We sample random mechanisms ![$f\sim P_F$](https://render.githubusercontent.com/render/math?math=%24f%5Csim%20P_F%24), where ![$F$](https://render.githubusercontent.com/render/math?math=%24F%24) corresponds to one of the following families
### Cubic Splines
![](./mechanism/cubic_spline.png?raw=true)

### Matern 2.5 Shift/Scale/Amplitude Sums
This name refers to a function of the form
![f(x) =  \sum_{i}^K a_i\cdot f_i\,(\mu_i + \sigma_i\cdot x)](https://render.githubusercontent.com/render/math?math=f(x)%20%3D%20%20%5Csum_%7Bi%7D%5EK%20a_i%5Ccdot%20f_i%5C%2C(%5Cmu_i%20%2B%20%5Csigma_i%5Ccdot%20x))
with all the parameters ![\{K\}\cup\{a_i,\mu_i,\sigma_i,\lambda_i\}_i](https://render.githubusercontent.com/render/math?math=%5C%7BK%5C%7D%5Ccup%5C%7Ba_i%2C%5Cmu_i%2C%5Csigma_i%2C%5Clambda_i%5C%7D_i) randomized (and ![\lambda_i](https://render.githubusercontent.com/render/math?math=%5Clambda_i) is the matern bandwidth)
![](./mechanism/matern_sums.png?raw=true)

### Sigmoid AM
Sigmoidal functions of the type ![x\mapsto \dfrac{b\cdot(x+c)}{1+\lvert b\cdot(x+c)\rvert}](https://render.githubusercontent.com/render/math?math=x%5Cmapsto%20%5Cdfrac%7Bb%5Ccdot(x%2Bc)%7D%7B1%2B%5Clvert%20b%5Ccdot(x%2Bc)%5Crvert%7D)
![](./mechanism/sigmoid_am.png?raw=true)

### Tanh Shift/Scale/Amplitude Sums
This name refers to a function of the form
![f(x) = \sum_{i}^K a_i\cdot f_i(\b_i\cdot(x+c_i))](https://render.githubusercontent.com/render/math?math=f(x)%20%3D%20%5Csum_%7Bi%7D%5EK%20a_i%5Ccdot%20f_i(%5Cb_i%5Ccdot(x%2Bc_i)))
with all the parameters ![\{K\}\cup\{a_i,\b_i,\c_i\}_i](https://render.githubusercontent.com/render/math?math=%5C%7BK%5C%7D%5Ccup%5C%7Ba_i%2C%5Cb_i%2C%5Cc_i%5C%7D_i) randomized.  

![](./mechanism/tanh_sum.png?raw=true)

# Acknowledgments
[This clever hack](https://alexanderrodin.com/github-latex-markdown/?math=(X%2CY)%20%5Csim%20P_%7BX%2CY%7D) allows one to change latex code to markdown-rendered images
