# Adapting the SLOPE algorithm to python (numpy & pytorch)

The slope algorithm carefully tests several basis functions one by one,
and then all together, finding the best compromise between goodness of fit and
function complexity as definded in [their paper](https://arxiv.org/pdf/1709.08915.pdf).

The original code is in R, we attempt to translate it into a pytorch/numpy compatible format.

## Examples of fitting procedure
The black is the true function, the blue is the generic fit (all basis functions together),
and the green each time is each basis function alone
### Toy Example, 13 basis functions

![](./slope_n_func_eq_13_two.png?raw=true)
