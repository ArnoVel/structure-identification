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
