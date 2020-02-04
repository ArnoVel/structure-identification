# Using differentiable GPs and differentiable HSIC to fit ANM-GPs

In this case, we are seeking to fit an additive model with **something else** than gaussian likelihood ('=' least squares).  
In particular, we are looking to solve ![\min_{\theta}\Big\lbrace \textrm{HSIC}_k\left(\textbf{X},\textbf{Y}-f(\textbf{X};\theta)\right) - \gamma\textrm{log}\,P\left(\textbf{Y}\mid \textbf{X};\theta\right)\Big\rbrace](https://render.githubusercontent.com/render/math?math=%5Cmin_%7B%5Ctheta%7D%5CBig%5Clbrace%20%5Ctextrm%7BHSIC%7D_k%5Cleft(%5Ctextbf%7BX%7D%2C%5Ctextbf%7BY%7D-f(%5Ctextbf%7BX%7D%3B%5Ctheta)%5Cright)%20-%20%5Cgamma%5Ctextrm%7Blog%7D%5C%2CP%5Cleft(%5Ctextbf%7BY%7D%5Cmid%20%5Ctextbf%7BX%7D%3B%5Ctheta%5Cright)%5CBig%5Crbrace)

We only draw 500 points to fit, due to finite sample limitations, we can find cases in which a standard gp fit "magically" finds dependence or independence while still giving the model a high marginal likelihood.

We include 3 plots each time:

* **prefit** the posterior mean and variance of a RBF GP without optimizing hyperparameters
* **postfit_nll** optimizing the hyperparameters using a classic marginal nll approach
* **postfit_hsic** optimizing (min.) HSIC(Res,X) where Res are residuals.  
  Two technical details are: we optimize HSIC + reg*NLL (stability, reg=0.1 usually),
  and we often warm start the second optimization using as init the outcome of **nll** fitting


## unintuitive examples

Examples where the p-value of the hsic-gamma test is very high (and so the test statistic really low compared to threshold) in some unintuitive scenarios:
* even before fitting the GP the pval is higher, and lower after nll/hsic fitting, but both fail to reject independence
* the pval is really low and we reject independence before training, but somehow training using NLL gives much higher pval compared to hsic training

### No NLL hyperparameter fitting
![](./unintuitive/one/prefit.png?raw=true)

### With NLL hyperparameter fitting
![](./unintuitive/one/postfit_nll.png?raw=true)

### With HSIC+NLL hyperparameter fitting
![](./unintuitive/one/postfit_hsic.png?raw=true)

### Conclusion

Although the initial hyperparameter are enough for the HSIC p-value to be greater than 0.05,
one can see the maximisation improves the likelihood of independence under HSIC.

## intuitive examples

Examples where the p-value of the hsic-gamma is intuitively low at first
* before fitting the GP, or after fitting. Barely passes the test at alpha = 0.05 without
* without hsic fitting, we wrongly reject the independence hypothesis, and hsic fitting 'solves' this.

### No NLL hyperparameter fitting
![](./intuitive/one/prefit.png?raw=true)

### With NLL hyperparameter fitting
![](.nintuitive/one/postfit_nll.png?raw=true)

### With HSIC+NLL hyperparameter fitting
![](./intuitive/one/postfit_hsic.png?raw=true)

### Conclusion

This is the best case scenario.
