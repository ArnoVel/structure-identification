import numpy as np

def exp_gauss(loc,scale,size=1000,alpha=1):
    vals = np.random.normal(0,1,size=size)
    vals = np.power(np.abs(vals),alpha)*np.sign(vals)
    vals = vals*scale + loc
    return vals

def unif(loc,scale,size=1000,alpha=1):
    vals = np.random.uniform(0,1,size=size)
    vals = vals*scale + loc
    return vals

def tri(loc,scale,size=1000,alpha=1):
    v1,v2 = np.random.uniform(0,1,size=size//2),\
            np.random.uniform(0,1,size=size//2)
    vals = v1+3*v2
    vals = vals*scale + loc
    return vals

def two_dists_mixed(N, mus=[-1,1], sigmas=[2,1], sampler=exp_gauss):
    v1,v2 = sampler(mus[0],sigmas[0],size=N//2, alpha=0.5),\
            sampler(mus[1],sigmas[1],size=N//2, alpha=1.2)
    x = np.concatenate((v1,v2))
    mu, sigma = x.mean(), x.std()
    x = (x-mu)/sigma
    return x
