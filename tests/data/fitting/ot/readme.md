# Fitting point clouds with underlying causal structure

## 'Basic' Gradient Flows
This does not yet fit a generative model.
Instead of optimizing over the parameters of a function which
produces point clouds conditioned on input data, only studies
the experimental performance of each gradient flow (behavior of the loss)
on specific ANM data.

## 'FCM' Gradient Flows
Instead of updating a point-cloud's position, generates conditional point clouds
and updates the model's parameters to minimize OT costs, without directly altering
point positions.
