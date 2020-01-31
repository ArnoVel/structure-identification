import torch
from functions.operations import _mvar_smooth_ecdf
import matplotlib.pyplot as plt
import seaborn as sns


# generate strange 2D data:
radius = 4
R,T = torch.normal(0,0.5,(600,)), torch.randn(600)
X,Y = (radius+R).cos()*T.cos() , (radius+R).cos()*T.sin()

sns.jointplot(X,Y,
              kind='scatter',
              ratio=5, height=7,
              space=0, alpha=0.3).plot_joint(sns.kdeplot,
                                             zorder=0,
                                             n_levels=8,
                                             alpha=0.4,
                                             color='blue')

plt.show()

Z = torch.stack([X,Y]).t()

Z_u = _mvar_smooth_ecdf(Z, x_vals=False); X_u, Y_u = Z_u[:,0], Z_u[:,1]

sns.jointplot(X_u,Y_u,
              kind='scatter',
              ratio=5, height=7,
              space=0, alpha=0.3).plot_joint(sns.kdeplot,
                                             zorder=0,
                                             n_levels=8,
                                             alpha=0.4,
                                             color='blue')

plt.show()
