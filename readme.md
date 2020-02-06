# Probing the Structure of Bivariate Distributions

The goal of this repository is to compare and compile a list of different statistical methods and algorithms,
which take as inputs bivariate data (point clouds) and attempt to infer a causal direction.

Good references on this topic are:

* [A Very Comprehensive Benchmark](http://jmlr.org/papers/volume17/14-518/14-518.pdf) of methods using Additive Noise Models, and all the surrounding concepts
* Several machine-learning using distribution embeddings have been designed: [RCC](https://arxiv.org/pdf/1409.4366.pdf), [KCDC](https://arxiv.org/pdf/1804.04622.pdf). A more statistical approach is [QCDC](https://arxiv.org/pdf/1801.10579.pdf) (copulas + quantile scores)
* [The SLOPE algorithm](https://arxiv.org/pdf/1709.08915.pdf) is a framework using a set of basis functions and which iteratively weight goodness of fit and
  function complexity. Various instantiations exist such as Slope-S, Slope-D, [an identifiable variant](https://eda.mmci.uni-saarland.de/pubs/2019/sloppy-marx,vreeken-wappendix.pdf), etc...
  More information can be found in [their journal paper](https://link.springer.com/article/10.1007/s10115-018-1286-7)
* [RECI](http://proceedings.mlr.press/v84/bloebaum18a/bloebaum18a.pdf) is a statistical approach based on regression, identifiable in the low noise setting
* [IGCI](https://staff.science.uva.nl/j.m.mooij/articles/ai2012.pdf) Justifies a statistical approach in the case the relationship is deterministic and invertible. Additional material can be found in [their subsequent paper](https://arxiv.org/pdf/1402.2499.pdf).
* A good [review on graphical models](https://www.frontiersin.org/articles/10.3389/fgene.2019.00524/full) for a number > 2 of variables can also be helpful to understand the general POV.
* [CGNN](https://arxiv.org/pdf/1709.05321.pdf) Connects graphical models, generative models, and bivariate methods in an interpretable fashion (using neural networks). It is a good bridge between bivariate and graph methods.


## Dependence - Independence Measures

Many causal algorithms rely on independence tests and Similarity tests. Some examples are

* Bivariate Methods using [Additive Noise Models](http://jmlr.org/papers/volume17/14-518/14-518.pdf) often use Mutual Information or HSIC
* Constraint-based methods for graph data use conditional independence tests. A good statistical test is the [KCI Test](https://arxiv.org/pdf/1202.3775.pdf) and the related KPC algorithm
* A good review on Dependence tests can be found in [this interesting thesis](https://arxiv.org/pdf/1607.03300.pdf)

Here we are interested in **differentiable** versions of various statistical tests. We implemented some tests using [PyTorch](https://pytorch.org/) and using smooth approximations to existing tests,
allowing backpropagation w.r.t each inputs/parameters.

### HSIC

* PyTorch [HSIC Test](dependence/hsic.py) and an example of [HSIC minimization](tests/data/gp/with_hsic/) ( [code](tests/test_gp.py) ) for ANM-detection.
  Although the HSIC test is differentiable wrt all inputs, our implementation doesn't yet support hyperparameter fitting.

* Examples of 2D gaussian HSIC-Gamma test, and ANM-detection tests will be uploaded.

* Might re-implement [relative HSIC between two models](https://arxiv.org/pdf/1406.3852.pdf)

### MMD

* PyTorch [MMD Test](dependence/mmd.py) with Gamma Approximation.

* Might re-implement optimized MMD from [here](https://github.com/dougalsutherland/opt-mmd) or [relative MMD between two models](https://arxiv.org/pdf/1511.04581.pdf)

## Bivariate Causal Algorithms

### SLOPE

We are currently [re-implementing SLOPE](causal/slope) in python, allowing both Numpy & PyTorch datatypes.
An example of the SLOPE fit for 13 basis functions can be found [in this folder](tests/data/fitting/slope) ( [code](tests/test_slope_fits.py) ).

## Distribution fittings

### Flexible Gaussian Mixtures

[Fit a GMM](tests/data/fitting/gmm) ( [code](test/test_gmm_fit.py) ) with flexible number of components.

* One dimensional on synthetic data (can be applied to estimate marginal complexity)
* Two dimensional on synthetic data (as an example of causality-agnostic distribution fitting)
