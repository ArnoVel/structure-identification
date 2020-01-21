import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X = np.linspace(-5,5,10000)

def exp_gauss(loc,scale,size=1000,alpha=1):
    vals = np.random.normal(0,1,size=size)
    vals = np.power(np.abs(vals),alpha)*np.sign(vals)
    vals = vals*scale + loc
    return vals


size = 10000
v1,v2 = exp_gauss(-1,2,size=size, alpha=0.5),\
        exp_gauss(3,1,size=size, alpha=1.2)
vals = np.concatenate((v1,v2))
mu, sigma = vals.mean(), vals.std()

v1,v2 = (v1-mu)/sigma, (v2-mu)/sigma
vals = (vals-mu)/sigma
plt.hist(vals, bins=2*size//50, alpha=0.4, color='royalblue', density=True)
plt.hist(v1, bins=size//50, alpha=0.1, color='r', density=True); plt.hist(v2, bins=size//50, alpha=0.1, color='g', density=True)
sns.rugplot(vals,color='k', lw=0.1, alpha=0.05, height=0.02); sns.kdeplot(vals);
plt.show()
