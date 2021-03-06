# Examples of SLOPE fitting procedure

This is a more detailed view, describing each fit in the process of finding the best SLOPE score realization.
The black is the true function, the blue is the generic fit (all basis functions together), and the green each time is each basis function alone.
## Toy Example, 13 basis functions
Example one:  
![](./slope_n_func_eq_13_one.png?raw=true)
Example two:  
![](./slope_n_func_eq_13_two.png?raw=true)

## Mixed Fits, 8 basis functions
Mixed fits cover all possible combinations of the list. We restrict the number of basis functions,
as 2^8 is already 256, and displaying more graphs using matplotlib becomes a difficult task.

The black line is the true function, the blue one is the generic fit, and the green one the corresponding
mixed fit. The title each time specifies which basis function was chosen. Examples:
`poly3` means `a*x^3`, while  `poly_inv2` means `a* 1/x^2`.  

![](./slope_mixed_func_nfunc_eq_8.png?raw=true)
