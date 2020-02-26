# Testing Classifier Two Sample Test

Check the [reference paper](https://arxiv.org/pdf/1610.06545.pdf) for theoretical analysis.
Potential experiments include bivariate causal discovery fits for ANMs, GNNs, GeometricNets (GNN using Geometric Losses such as Sinkhorn/OT).
Code is not available yet, we provide short snippets if possible.

## Simple 2D Gaussian experiments

Distinguish means of gaussians (0 vs `m` ranging from 0.1 to 1).  

In this case, `m=1` is easily distringuished by both classifiers, with accuracy 75% leading to `pval=0`. The biggest gap for which this happens is `m=0.5`, and at `m=0.4` the values are non-zero but NN has a pval of approx `1e-07` and KNN `1e-03`.
* at `m=0.3`, NN has pval `1e-05` and KNN `1e-02` (passes at `alpha=0.01`)
* at `m=0.2`, NN has pval `0.03` and KNN `0.80` (both pass at `alpha=0.01`)

etc ...

Can be reproduced using `c2st.py` through

```python
for mu in torch.linspace(0,1,10):
        P,Q = torch.randn(n,2), torch.randn(n,2)+mu*torch.ones(1,2)
        print(c2st.knn_c2st(P,Q))
        print(c2st.neural_net_c2st(P,Q))
        print(f'end of loop for mu={mu}')
```

## ANM Causal Sinkhorn fit using CS2T as Goodness-of-Fit Criterion

More details will be given in the future, but sampling causal data using `functions/generators/generators.py`, and fitting causal and anticausal models, we can then compare how similar they are to the origin point cloud.
In this case, we sample from a random ANM using `random.choice` with seed set to 22, getting the triplet `('subgmm', 'student', 'spline')`.

Running sinkhorn L1 loss minimization with a two layer network, we obtain the fits for `X --> Y` as

![](./anm/sinkhorn_fit_n_1000_xy.png?raw=true)

along with C2ST values
```
C2ST test for XY = XY_hat
c2st neural test: acc=0.4960000216960907, P(T>acc)=0.5998585224151611,  (reject if pval < 1e-02)
c2st knn test: acc=0.46800002455711365, P(T>acc)=0.9785075187683105,  (reject if pval < 1e-02)
```
and the fits for `Y --> X` as

![](./anm/sinkhorn_fit_n_1000_yx.png?raw=true)

along with C2ST values

```
C2ST test for XY = XY_hat
c2st neural test: acc=0.5210000276565552, P(T>acc)=0.09206289052963257,  (reject if pval < 1e-02)
c2st knn test: acc=0.4780000150203705, P(T>acc)=0.9179481267929077,  (reject if pval < 1e-02)
```

We can see that causal and anticausal models are extremely likely in terms of KNN tests, while NN two sample testing. While in both cases the causal model is identified, it is only by a small margin (ratio of 0.93) for KNN, and large (ratio of 6.5) for the NN instantiation.
