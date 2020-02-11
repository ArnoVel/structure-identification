# Adapting the SLOPE algorithm to python (numpy & pytorch)

The slope algorithm carefully tests several basis functions one by one,
and then all together, finding the best compromise between goodness of fit and
function complexity as defined in [their paper](https://arxiv.org/pdf/1709.08915.pdf).

The original code is in R, we attempt to translate it into a pytorch/numpy compatible format.

## Finding the best set of basis functions
The two main **global** (a unique function f_g for the whole domain) approaches given a set of basis functions are:

* Either find the 'best' (in terms of score) individual function so that `y_i = a*f(x_i) + b` (GLM for `f`).
  This method is called `fitI` or in our implementation `_fit_index`

* Either find the 'best' subset by enumerating all possibilities. A set of more than 13 functions is not recommended,
  as 13 functions already means fitting 8191 models using least squares on the samples.

* Or least squares using the full set of basis functions, obtain best fit regardless of complexity score.
  One could compare sparsity induced by a classic Lasso regression tactic versus sparsity obtained through SLOPE scoring,
  but we haven't implemented this yet.

Below we give two examples of the best-fit search, and then more examples giving detail of each fits.
The functions used in the example are of the type `y = sin(x)*sigmoid(x)*((x-1)^pow1)/5 - (x+1)^pow2)/5)`, where `pow1,pow2` are integer powers.

### Example in the low-noise setting



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

As an example, using 13 functions (setting both torch & numpy seeds as 1020), on the following data

![](./slope_generic.png?raw=true)

* The scores in increasing order are
  ```
  6184.2236 6201.253 6201.4907 6203.2 6207.377 6208.0234
  ...
  6443.616 6443.7754 6444.083 6444.7114 6445.641 6466.3438`
  ```
* corresponding to basis functions
  ```
  'exp',
  'poly2',
  'poly1',
  'poly1+poly2',
  'exp+poly2',
  'poly4',
  ...
  'poly0+exp+poly1+poly2+poly3+poly5+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5',
  'exp+poly1+poly2+poly3+poly4+poly5+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5',
  'poly0+exp+poly2+poly3+poly4+poly5+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5',
  'poly0+exp+poly1+poly2+poly4+poly5+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5',
  'poly0+poly1+poly2+poly3+poly4+poly5+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5',
  'poly0+exp+poly1+poly2+poly3+poly4+poly5+poly_inv1+poly_inv2+poly_inv3+poly_inv4+poly_inv5'
  ```
The lists are much longer, but only using the top 6 and bottom 6 value, one learns that

* sparsity is preferred, but using more than 1 basis functions can yield similar/better scores when needed
* Using a very high number of basis functions isn't preferred (the best use 1-2, which is << 13)
* polynomials and exponential components tend to give high score: good compromise complexity/goodness of fit
* Using a high number of inverse polynomials ('complex functions') is penalized, but a small number of them is beneficial.
  **for example:** at the ranks 9-17, one finds 9 inverse polynomials, either alone or with exponential component,
  * the scores range from 6209.575 to 6211.8154,
  * comparable to linear fits in score,
  * very close to best ranks, and far away from average (~ 6280)
  * the first rank at more than 2 inverse polynomials is 367 with score 6259.1533, much closer to average

This is **obviously** data-depedent, but the chosen function was a polynomial multiplied with a sigmoid and a sinusoid (relatively simple).
