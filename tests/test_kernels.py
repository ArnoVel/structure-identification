import torch
from functions.kernels import SumIdentical, RBF, RQ

sampler = torch.distributions.normal.Normal(loc=0,scale=1)

n = int(1e04)
X = sampler.sample(sample_shape=(n,2))

device = torch.device('cuda')
print(device)
X = X.to(device)

params = [{'bandwidth':10**(i),
            'alpha':(4+i)
            } for i in range(-2,2)]
SumK = SumIdentical(params=params, kernel=RQ).to(device)

print(SumK(X))
