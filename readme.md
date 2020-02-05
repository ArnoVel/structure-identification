# Probing the Structure of Bivariate Distributions

The goal of this repository is to compare and compile a list of different statistical methods and algorithm,
which take as inputs bivariate data (point clouds) and attempt to infer a causal direction.

Good references on this topic are:

* [A Very Comprehensive Benchmark](http://jmlr.org/papers/volume17/14-518/14-518.pdf) of methods using Additive Noise Models, and all the surrounding concepts
* [The SLOPE algorithm](https://arxiv.org/pdf/1709.08915.pdf) is a framework using a set of basis functions and which iteratively weight goodness of fit and
  function complexity. Various instantiations exist such as Slope-S, Slope-D, [an identifiable variant](https://eda.mmci.uni-saarland.de/pubs/2019/sloppy-marx,vreeken-wappendix.pdf), etc...
  More information can be found in [their journal paper](https://link.springer.com/article/10.1007/s10115-018-1286-7)
* A good [review on graphical models](https://www.frontiersin.org/articles/10.3389/fgene.2019.00524/full) for a number > 2 of variables can also be helpful to understand the general POV.
