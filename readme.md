# Probing the Structure of Bivariate Distributions

The goal of this repository is to compare and compile a list of different statistical methods and algorithms,
which take as inputs bivariate data (point clouds) and attempt to infer a causal direction.

Good references on this topic are:

* [A Very Comprehensive Benchmark](http://jmlr.org/papers/volume17/14-518/14-518.pdf) of methods using Additive Noise Models, and all the surrounding concepts
* [The SLOPE algorithm](https://arxiv.org/pdf/1709.08915.pdf) is a framework using a set of basis functions and which iteratively weight goodness of fit and
  function complexity. Various instantiations exist such as Slope-S, Slope-D, [an identifiable variant](https://eda.mmci.uni-saarland.de/pubs/2019/sloppy-marx,vreeken-wappendix.pdf), etc...
  More information can be found in [their journal paper](https://link.springer.com/article/10.1007/s10115-018-1286-7)
* A good [review on graphical models](https://www.frontiersin.org/articles/10.3389/fgene.2019.00524/full) for a number > 2 of variables can also be helpful to understand the general POV.


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

### MMD

* PyTorch [MMD Test](dependence/mmd.py) with Gamma Approximation.

## Bivariate Causal Algorithms

### SLOPE

We are currently [re-implementing SLOPE](causal/slope) in python, allowing both Numpy & PyTorch datatypes.
An example of the SLOPE fit for 13 basis functions can be found [in this folder](tests/data/fitting/slope) ( [code](tests/test_slope_fits.py) ).
