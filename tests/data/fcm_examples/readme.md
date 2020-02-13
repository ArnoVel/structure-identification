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

### RBF GP Randomized Quantile sums
This name refers to a function of the form  
![\frac{1}{5}\sum_{i=1}^5 a_i\cdot f( \cdot ; \alpha_i)](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B1%7D%7B5%7D%5Csum_%7Bi%3D1%7D%5E5%20a_i%5Ccdot%20f(%20%5Ccdot%20%3B%20%5Calpha_i))  

where ![f( \cdot ; \alpha_i)](https://render.githubusercontent.com/render/math?math=f(%20%5Ccdot%20%3B%20%5Calpha_i)) is the ![\alpha_i](https://render.githubusercontent.com/render/math?math=%5Calpha_i) -th quantile of a random GP fit to a random function using k random points with all the parameters ![\{a_i,\alpha_i\}_i](https://render.githubusercontent.com/render/math?math=%5C%7Ba_i%2C%5Calpha_i%5C%7D_i) randomized. The bandwidths of the GPs are randomly set using a folded normal.

![](./mechanism/rbfgp_sums.png?raw=true)

### Sigmoid AM
Sigmoidal functions of the type ![x\mapsto \dfrac{b\cdot(x+c)}{1+\lvert b\cdot(x+c)\rvert}](https://render.githubusercontent.com/render/math?math=x%5Cmapsto%20%5Cdfrac%7Bb%5Ccdot(x%2Bc)%7D%7B1%2B%5Clvert%20b%5Ccdot(x%2Bc)%5Crvert%7D)
![](./mechanism/sigmoid_am.png?raw=true)

### Tanh Shift/Scale/Amplitude Sums
This name refers to a function of the form
![f(x) = \sum_{i}^K a_i\cdot f_i(\b_i\cdot(x+c_i))](https://render.githubusercontent.com/render/math?math=f(x)%20%3D%20%5Csum_%7Bi%7D%5EK%20a_i%5Ccdot%20f_i(%5Cb_i%5Ccdot(x%2Bc_i)))
with all the parameters ![\{K\}\cup\{a_i,\b_i,\c_i\}_i](https://render.githubusercontent.com/render/math?math=%5C%7BK%5C%7D%5Ccup%5C%7Ba_i%2C%5Cb_i%2C%5Cc_i%5C%7D_i) randomized.  

![](./mechanism/tanh_sum.png?raw=true)


## Noise Distributions
To be added, we count any symmetric distribution for which mean = mode = 0 as a potential noise source for a model `Y = f(X) + N` if anm, or `Y = f(X)+ N*g(X)` if heteroskedastic.

## Resulting point clouds
To list every combination, we would need 288 different types of causal distributions.
One can find a find examples [in this folder](./pairs). The full gallery can be found [in my pictures-only repository](https://github.com/ArnoVel/causal-pictures).
For example,
### ANM case
![](./pairs/anm_True_c_subgmm_bn_student_m_sigmoidam.png?raw=true)
### HTR case
![](./pairs/anm_False_c_gmm_bn_beta_m_sigmoidam.png?raw=true)

# Acknowledgments
[This clever hack](https://alexanderrodin.com/github-latex-markdown/?math=(X%2CY)%20%5Csim%20P_%7BX%2CY%7D) allows one to change latex code to markdown-rendered images
