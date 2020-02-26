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
* [CGNN](https://arxiv.org/pdf/1709.05321.pdf) Connects graphical models, generative models, and bivariate methods in an interpretable fashion (using neural networks). It is a good bridge between bivariate and graph methods. The authors are currently building [a very helpful python causal discovery library](https://github.com/FenTechSolutions/CausalDiscoveryToolbox)


## Dependence - Independence Measures

Many causal algorithms rely on independence tests and Similarity tests. Some examples are

* Bivariate Methods using [Additive Noise Models](http://jmlr.org/papers/volume17/14-518/14-518.pdf) often use Mutual Information or HSIC
* Constraint-based methods for graph data use conditional independence tests. A good statistical test is the [KCI Test](https://arxiv.org/pdf/1202.3775.pdf) and the related KPC algorithm.
  In case one needs a faster, approximate method, the authors (and others) have recently [designed approximations](https://arxiv.org/pdf/1702.03877.pdf) such as RCIT and RCOT.
  Another good but quadratic complexity conditional independence test is [PCIT](http://auai.org/uai2014/proceedings/individuals/194.pdf)
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

### C2ST
Classifier Two Sample Tests (C2ST) have been introduced and tested [in this paper](https://arxiv.org/pdf/1610.06545.pdf). Here, we [re-implement](dependence/c2st) and slightly adapt the lua code of the authors, which includes
* C2ST-NN: using a shallow neural network classifier (ReLU + Sigmoid) with default 20 hidden units.
  While adding layers/hidden units is a good idea, we usually work with 500-5000 samples per distribution, and/or aim for accuracy higher than 55% to reject P=Q
* C2ST-KNN: K-nearest neighbors classifier with `k=floor(n_te/2)`. Usually worse than neural nets.

The idea in broad terms is that under H0 (P=Q) , the classifier cannot exceed 50% accuracy and `n*acc` is distributed as `Binomial(n_te, 0.5)`. Then `acc` under H0 can be approximated as `Normal(0.5, 0.25/n_te)`, we therefore use the approximate null to find a p-value on the accuracy and reject H0 accordingly.  
Some basic examples can be found [in this subdirectory](tests/data/c2st).

## Bivariate Causal Algorithms

### SLOPE

We are currently [re-implementing SLOPE](causal/slope) in python, allowing both Numpy & PyTorch datatypes.
An example of the SLOPE fit for 13 basis functions can be found [in this folder](tests/data/fitting/slope) ( [code](tests/test_slope_fits.py) ), which also contains mixed fits for 8 functions, and a little bit more.

## Distribution fittings

### Flexible Gaussian Mixtures

[Fit a GMM](tests/data/fitting/gmm) ( [code](test/test_gmm_fit.py) ) with flexible number of components.

* One dimensional on synthetic data (can be applied to estimate marginal complexity)
* Two dimensional on synthetic data (as an example of causality-agnostic distribution fitting)

## Experiments and Visualisations
Unless exceptions, every picture and experiment reported can be seen in [the tests/data subdir](./tests/data).
However, for particularly large files or high number of pictures, a [different picture-only repo](https://github.com/ArnoVel/causal-pictures) is available!  

The dependencies can be installed using `pip install -r requirements.txt` or
`pip3 install -r requirements.txt`
