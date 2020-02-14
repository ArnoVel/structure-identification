import numpy as np
import scipy as sp
import torch
from scipy.stats import semicircular
from functions.regressors.gp.models import GaussianProcess
from functions.regressors.gp.kernels import RBFKernel, WhiteNoiseKernel
import scipy.interpolate as itpl
import numpy.random as rd
import pandas as pd
import h5py

class  CauseSampler(object):
    """General class to generate a cause variable X"""
    def __init__(self, sample_size=1000):
        super(CauseSampler, self).__init__()
        self.n = sample_size

    def uniform(self):
        lb, ub = rd.normal(0,3,2)
        lb, ub = min(lb,ub), max(lb,ub)
        cause = rd.uniform(lb,ub,self.n)
        cause = (cause - cause.mean())/cause.std()
        return cause

    def uniform_mixture(self, n_classes=None):
        if not n_classes:
            n_classes = rd.randint(1,6)

        s1, s2 = rd.uniform(0,5,2)

        means = rd.normal(0,s1, size=n_classes)
        stds = np.abs(rd.normal(0,s2, size=n_classes))

        mixt_weights = np.abs(rd.normal(0,1, size=n_classes))
        mixt_weights = mixt_weights / sum(mixt_weights)

        cause = _unif_mixture_sampler(self.n, mixt_weights, means, stds)
        cause = (cause - cause.mean())/cause.std()
        return cause

    def gaussian_mixture(self, n_classes=None):
        """ fixed conditional mean and variance distributions,
            given mean, variance each class distribution is normal
            """
        if not n_classes:
            n_classes = rd.randint(1,6)

        s1, s2 = rd.uniform(0,5,2)
        means = rd.normal(0,s1, size=n_classes)
        stds = np.abs(rd.normal(0,s2, size=n_classes))

        mixt_weights = np.abs(rd.normal(0,1, size=n_classes))
        mixt_weights = mixt_weights / sum(mixt_weights)

        cause = _gauss_mixture_sampler(self.n, mixt_weights, means, stds)
        cause = (cause - cause.mean())/cause.std()
        return cause

    def subgaussian_mixture(self, n_classes=None):
        """ fixed hyperparameter prior on the subgaussian type,
        we use the common method sign(x)*abs(x)^q where x gaussian , 0<q<1
        """
        if not n_classes:
            n_classes = rd.randint(1,6)
        subgaussian_pow = [0.3+0.1*i for i in range(n_classes)]

        # generate shape and scale parameters of subgaussians from gaussians
        s1, s2 = rd.uniform(0,5,2)
        means = rd.normal(0,s1, size=n_classes)
        stds = np.abs(rd.normal(0,s2, size=n_classes))

        mixt_weights = np.abs(rd.normal(0,1, size=n_classes))
        mixt_weights = mixt_weights / sum(mixt_weights)

        cause = _powgauss_mixture_sampler(self.n, mixt_weights,
                                        powers=subgaussian_pow,
                                        means=means, stds=stds)
        cause = (cause - cause.mean())/cause.std()

        return cause

    def supergaussian_mixture(self, n_classes=None):
        """ fixed hyperparameter prior on the supergaussian type,
        we use the common method sign(x)*abs(x)^q where x gaussian , q>1
        """
        if not n_classes:
            n_classes = rd.randint(1,6)
        pow_vals_ = np.linspace(0,0.7,n_classes) + 1.05

        # generate shape and scale parameters of subgaussians from gaussians
        s1, s2 = rd.uniform(0,5,2)
        means = rd.normal(0,s1, size=n_classes)
        stds = np.abs(rd.normal(0,s2, size=n_classes))

        mixt_weights = np.abs(rd.normal(0,1, size=n_classes))
        mixt_weights = mixt_weights / sum(mixt_weights)

        cause = _powgauss_mixture_sampler(self.n, mixt_weights,
                                        powers=pow_vals_,
                                        means=means, stds=stds)
        cause = (cause - cause.mean())/cause.std()

        return cause

    def subsupgaussian_mixture(self, n_classes=None):
        """ fixed hyperparameter prior on the super or sub gaussian type,
        we use the common method sign(x)*abs(x)^q where x gaussian , q<0
        """
        if not n_classes:
            n_classes = rd.randint(1,6)
        # base powers from 0.3 to 2.3
        pow_vals_ = np.linspace(0,1.4,20) + 0.3

        #randomly sample n_classes powers from this array
        powers = rd.choice(pow_vals_,size=n_classes,replace=False)
        # generate shape and scale parameters of subgaussians from gaussians
        s1, s2 = rd.uniform(0,5,2)
        means = rd.normal(0,s1, size=n_classes)
        stds = np.abs(rd.normal(0,s2, size=n_classes))

        mixt_weights = np.abs(rd.normal(0,1, size=n_classes))
        mixt_weights = mixt_weights / sum(mixt_weights)

        cause = _powgauss_mixture_sampler(self.n, mixt_weights,
                                        powers=powers,
                                        means=means, stds=stds)
        cause = (cause - cause.mean())/cause.std()
        return cause


class MechanismSampler(object):
    """Given only partial information about X (such as domain),
       randomly sample a causal mechanism X-->Y.
       This class is only about deterministic mechanisms"""
    def __init__(self, cause_sample):
        super(MechanismSampler, self).__init__()
        self.x = cause_sample

    def CubicSpline(self, n_knots=None):
        """ define hermite cubic spline
        with domain set to [min(x) - std(x) , max(x)+std(x)]
        Args:
        cause (np.array): the cause sample,  mean, unit variance
        """
        if not n_knots:
            n_knots = rd.randint(4,7)

        knots_y = rd.normal(0,1,n_knots)
        # get hermit cubic spline interpolator
        dom = np.linspace(np.min(self.x)-np.std(self.x),
                        np.max(self.x)+np.std(self.x),
                        num=n_knots)

        return itpl.CubicSpline(dom,knots_y)

    def SigmoidAM(self):
        a = rd.exponential(4) + 1
        if rd.rand()>.5:
            b = rd.uniform(0.5,2)
        else:
            b = - rd.uniform(0.5,2)
        c = rd.uniform(-2,2)

        return (lambda x:
                    b*(x+c)/(1+np.abs(b*(x+c))))

    def RbfGP(self, bounds=(5,20)):
        ''' draws from gp priors Matern 2.5,
            length scales from [4,10] for regularity
        '''
        nu = 2.5 ; #ls = rd.uniform(*bounds,3)
        #self._ls = ls
        ker = RBFKernel()
        gp = GaussianProcess(ker, eps=1e-02)
        n_knots = np.random.randint(100,300)
        x_knots = torch.linspace(np.min(self.x)-np.std(self.x),
                        np.max(self.x)+np.std(self.x),
                        n_knots)
        y = torch.normal(0,0.1,x_knots.shape);
        means, scales = torch.normal(0,5,(3,)), torch.normal(2,5,(3,)).abs()
        y_m = (((x_knots - means[0])*scales[0]).pow(2) +
                ((x_knots - means[1])*scales[1]).pow(3) +
                ((x_knots - means[2])*scales[2]).pow(4)
                ) * x_knots.sin()
        y = y + y.sin() + y_m
        x_knots, y = x_knots.view(-1,1), y.view(-1,1)
        gp.set_data(x_knots,y, reg=1e-02)
        #locs, scales = rd.normal(0,5,(len(gps),)), np.abs(rd.normal(0,5,(len(gps),)))


        def fun(X):
            if isinstance(X,torch.Tensor) and X.ndim <2:
                X_ = X.view(-1,1)
            if isinstance(X,np.ndarray):
                X_ = torch.from_numpy(X).view(-1,1)
                mean,var = gp(X_, return_var=True)
                mean, var = mean.detach().cpu(), var.detach().cpu()
                al = torch.rand(5)
                normal = torch.distributions.normal.Normal(mean,var)
                q = normal.icdf(al)
                q = (q * torch.normal(0,1,(1,5)).expand_as(q)).mean(1)
            return q.numpy()
        return fun

    def tanhSum(self):
        num_tanhs = np.random.randint(3,10)
        s1,s2,s3 = np.abs(np.random.normal(0,5,3))
        a,b,c = np.abs(0.5+np.random.normal(0,s1,num_tanhs)),\
                np.abs(0.5+np.random.normal(0,s2,num_tanhs)),\
                np.random.normal(0,s3,num_tanhs)
        def fun(X):
            d = X.reshape(-1,1) + c.reshape(1,-1)
            # once we have [N,num_tanhs] , the broadcasting will be automatic
            return (np.tanh(b*d)*a).sum(1) + X*1e-02
        return fun

class NoiseSampler(object):
    """ Given X and Y, generates noise N,
        assumes X,Y both scaled to unit variance, zero mean"""
    def __init__(self, cause_sample, effect_sample, anm=True, base_noise='normal'):
        self.cause_sample = cause_sample
        self.effect_sample = effect_sample
        self.anm = anm # if false heteroskedastic
        self.n = len(cause_sample)
        self.base_noise = base_noise

        noise_list = ['normal', 'student', 'triangular', 'uniform',
                       'beta', 'semicircular']
        if base_noise not in noise_list:
            raise NotImplementedError(f"base noise requested ({base_noise}) not in list: {noise_list}")

        if not anm:
            mech_sampler = MechanismSampler(self.cause_sample)
            self.noise_f = mech_sampler.CubicSpline()
        assert self.n == len(effect_sample)

    def add_noise(self):
        gamma = rd.uniform(0.3,0.7,1) # controls noise to effect ratio

        if self.base_noise == 'normal':
            base_noise_sample = rd.normal(0,1, size=self.n)
        elif self.base_noise == 'uniform':
            base_noise_sample = rd.uniform(-1,1, size=self.n)
        elif self.base_noise == 'triangular':
            base_noise_sample = rd.triangular(-1,0,1, size=self.n)
        elif self.base_noise == 'student':
            nu = rd.randint(4,8,1)
            base_noise_sample = rd.standard_t(nu, size=self.n)
        elif self.base_noise == 'beta':
            al = rd.randint(2,6,1)
            # similar to pert distribution
            base_noise_sample = rd.beta(al,al, size=self.n)*2 - 0.5
        elif self.base_noise == 'semicircular':
            # same as beta with 3/2 al, and centered at 0
            base_noise_sample = semicircular.rvs(size=self.n) * 0.8 # too big var apparently..

        if self.anm:
            effect = self.effect_sample
            # normalize to account for small sample deviations
            base_noise_sample = (base_noise_sample - base_noise_sample.mean()) / base_noise_sample.std()

            idx = np.argsort(self.cause_sample)
            y = gamma * effect + (1-gamma)*base_noise_sample

        else:
            effect = self.effect_sample
            # normalize to account for small sample deviations, but after mapping X
            het_noise = base_noise_sample * self.noise_f(self.cause_sample)
            het_noise = (het_noise - het_noise.mean()) / het_noise.std()

            y = gamma * effect + (1-gamma) * het_noise

        y = (y - y.mean()) / y.std()
        return y


class PairSampler(object):
    """ Materializes a bivariate FCM distribution:
        Once one sets the type of noise,cause,mechanism,
        One can call a succession of
        generate_cause --> fit_mechanism --> generate_effect --> generate_pair
        to obtain a new bivariate sample following the same
        FCM distribution with fixed sample size.
        Failure to call each of these methods one after the other
        will lead to nonsensical samples
    """

    def __init__(self, n, anm=True, base_noise='normal', cause_type='gmm', mechanism_type='spline'):
        super(PairSampler, self).__init__()
        self.n = n
        self.mechanism_type = mechanism_type
        self.base_noise = base_noise
        self.anm = anm

        cause_sampler = CauseSampler(sample_size=n)
        cause_choice = {
                        'gmm': cause_sampler.gaussian_mixture,
                        'subgmm': cause_sampler.subgaussian_mixture,
                        'supgmm': cause_sampler.supergaussian_mixture,
                        'subsupgmm': cause_sampler.subsupgaussian_mixture,
                        'uniform': cause_sampler.uniform,
                        'mixtunif': cause_sampler.uniform_mixture,
                        }
        try:
            self.cause_gen = cause_choice[cause_type]
        except Exception as e:
            raise NotImplementedError("cause distribution not implemented: ",cause_type)
        self.mechanism = None
        self.cause_sample = []
        self.effect_sample = []

    def fit_mechanism(self):
        assert len(self.cause_sample)>0
        mech_sampler = MechanismSampler(self.cause_sample)
        if self.mechanism_type == 'spline':
            self.mechanism = mech_sampler.CubicSpline()
        elif self.mechanism_type == 'sigmoidam':
            self.mechanism = mech_sampler.SigmoidAM()
        elif self.mechanism_type == 'tanhsum':
            self.mechanism = mech_sampler.tanhSum()
        elif self.mechanism_type == 'rbfgp':
            self.mechanism = mech_sampler.RbfGP()
        else:
            raise NotImplementedError("This mechanism type was not expected",self.mechanism_type)

    def generate_cause(self):
        self.cause_sample = self.cause_gen()

    def generate_effect(self):
        self.effect_sample = self.mechanism(self.cause_sample)
        effect = self.effect_sample ; self.effect_sample = (effect - effect.mean()) / effect.std()

    def generate_pair(self):
        noise_sampler = NoiseSampler(cause_sample= self.cause_sample,
                                     effect_sample= self.effect_sample,
                                     anm= self.anm,
                                     base_noise=self.base_noise)
        noisy_effect = noise_sampler.add_noise()
        self.noisy_effect = noisy_effect
        return np.vstack((self.cause_sample, noisy_effect))

    def get_new_pair(self):
        """ assumes the class init was called correctly. """
        self.generate_cause()
        self.fit_mechanism()
        self.generate_effect()
        pair = self.generate_pair()
        return pair


class DatasetSampler(object):
    def __init__(self, N, n=1000,
                anm = True,
                base_noise='normal',
                cause_type='gmm',
                mechanism_type='spline',
                with_labels=False):
        super(DatasetSampler, self).__init__()
        self.N = N
        self.n = n
        self.anm = anm
        self.pSampler = PairSampler(n,
                                anm=self.anm,
                                base_noise=base_noise,
                                cause_type=cause_type,
                                mechanism_type=mechanism_type)
        self.anm = anm
        self.mechanism_type = mechanism_type
        self.base_noise = base_noise
        self.with_labels = with_labels

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index <= self.N-1:
            pair = self.pSampler.get_new_pair()
            self.index += 1
            if self.with_labels:
                return pair, 1
            else:
                return pair
        else:
            raise StopIteration



def _unif_mixture_sampler(n,weights,means,stds):
    unif_choice = rd.choice(a=len(weights),
                                size=n,
                                p=weights)
    mixt_sample = np.array([rd.uniform(means[i]-stds[i],means[i]+stds[i]) for i in unif_choice]).ravel()
    return mixt_sample


def _gauss_mixture_sampler(n,weights,means,stds):

    gaussian_choice = rd.choice(a=len(weights),
                                size=n,
                                p=weights)
    mixt_sample = np.array([ rd.normal(means[i],stds[i]) for i in gaussian_choice]).ravel()

    return mixt_sample

def _powgauss_mixture_sampler(n,weights,powers,means,stds):

    subgauss_choice = rd.choice(a=len(weights),
                                size=n,
                                p=weights)
    mixt_sample = []

    for i in subgauss_choice:
        rnorm_sample = rd.normal(0,1)
        # get centered unit shape subgaussian
        subgauss_sample = np.sign(rnorm_sample)*np.power(np.abs(rnorm_sample),powers[i])
        # reposition and reshape
        subgauss_sample = subgauss_sample*stds[i]+means[i]
        mixt_sample.append(subgauss_sample)

    mixt_sample = np.array(mixt_sample).ravel()

    return mixt_sample


def _symmetrize_dataset(dataset):
    """ symmetrizes the cause-effect data,
    with rows of size (2,n) , and for each (x,y)
    swaps to (y,x) and add labels 1,-1
    Args:
        dataset: np.array of shape (N,2,n)
    Returns:
        sym_dataset: np.array of shape (2N,2,n)
        labels: np.array of size 2N
    """
    labels = []
    sym_pairs = []
    for row in dataset:
        x,y = row
        sym_pairs.append([x,y])
        sym_pairs.append([y,x])
        labels.append(1)
        labels.append(-1)
    return np.array(sym_pairs), np.array(labels)


def _to_dataframe(dataset):
    """ puts all the synthetic pairs into
    CEP dataframe format, as seen in the CDT
    package"""
    df_synth = pd.DataFrame({
    'A': [row for row in dataset[:,0,:]],
    'B': [row for row in dataset[:,1,:]]
    })

    return df_synth

def _check_file_extension(filepath):
    fname = filepath.split('/')[-1]
    if '.npy' in fname:
        return '.npy'
    elif '.h5' in fname:
        return '.h5'
    else:
        ValueError("this extension was not expected",fname)

def _load_wrapper(filepath):
    extension = _check_file_extension(filepath)
    if extension == '.npy':
        synth_te = np.load(filepath)
        synth_l_te = np.load(filepath.replace('pairs','labels'))
        return synth_te, synth_l_te
    elif extension == '.h5':
        f = h5py.File(filepath, 'r')
        return f
    else:
        NotImplementedError("This type of validation data is not expected",extension)
