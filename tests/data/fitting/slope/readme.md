# Adapting the SLOPE algorithm to python (numpy & pytorch)

The slope algorithm carefully tests several basis functions one by one,
and then all together, finding the best compromise between goodness of fit and
function complexity as definded in [their paper](https://arxiv.org/pdf/1709.08915.pdf).

The original code is in R, we attempt to translate it into a pytorch/numpy compatible format.

## Examples of fitting procedure
The black is the true function, the blue is the generic fit (all basis functions together),
and the green each time is each basis function alone
### Toy Example, 13 basis functions
Example one:  
![](./slope_n_func_eq_13_one.png?raw=true)
Example two:  
![](./slope_n_func_eq_13_two.png?raw=true)

### Mixed Fits, 8 basis functions
Mixed fits cover all possible combinations of the list. We restrict the number of basis functions,
as 2^8 is already 256, and displaying more graphs using matplotlib becomes a difficult task.

The black line is the true function, the blue one is the generic fit, and the green one the corresponding
mixed fit. The title each time specifies which basis function was chosen. Examples:
`poly3` means `a*x^3`, while  `poly_inv2` means `a* 1/x^2`.  

![](./slope_mixed_func_nfunc_eq_8.png?raw=true)

### Example for 13 mixed fits

It becomes too hard to plot, however a detailed report of the outputs of the fitting procedures on an example [can be found here](./res.out).
Using the data from each mixed fit, one can rank the models using the SLOPE scoring method.  

As an example, using 13 functions

* The scores in increasing order are `6820.789  6843.336  6844.5376 ... 8631.703  8632.497  8642.965`
* corresponding to basis functions
  ```
  'poly0+exp+poly1+poly2+poly3+poly4+poly5',
  'poly0+exp+poly1+poly2+poly3+poly4+poly5+poly_inv5',
  'poly0+exp+poly1+poly2+poly3+poly4+poly5+poly_inv1',
  ...
  'poly4+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5',
  'poly0+exp+poly2+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5',
  'poly0+exp+poly3+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5'

  ```
The lists are much longer, but only using the top 3 and bottom 3 value, one learns that

* Extreme sparsity is not preferred
* Using an very high number of basis functions isn't preferred (the best use 7-8, much less than 13)
* polynomials and exponential components tend to give high score: good compromise complexity/goodness of fit

This is data-depedent, but the chosen function was a polynomial multiplied with a sigmoid and a sinusoid (relatively simple).
