import numpy as np
import torch as th
from tqdm import trange
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import scale
from model import PairwiseModel
from loss import MMDloss
from scipy import integrate as itg
from scipy.stats.mstats import hdquantiles

from Settings import SETTINGS
from parallel import parallel_run
from pandas import DataFrame, Series
import os

# courtesy of https://github.com/Diviyan-Kalainathan/

class MmdNet(th.nn.Module):
    """ basic FCM approximation through MMD.
        Only models y|x as  Y = f(X,N ; theta)
    """
    def __init__(self, batch_size, nh=20, idx=0, verbose=None,
                 dataloader_workers=0, **kwargs):
        """Build the Torch graph.
        :param batch_size: size of the batch going to be fed to the model
        :param kwargs: h_layer_dim=(CGNN_SETTINGS.h_layer_dim)
                       Number of units in the hidden layer
        :param device: device on with the algorithm is going to be run on.
        """
        super(MmdNet, self).__init__()
        self.register_buffer('noise', th.Tensor(batch_size, 1))
        self.criterion = MMDloss(input_size=batch_size, bandwidths=None)
        self.layers = th.nn.Sequential(th.nn.Linear(2,nh),
                                    #th.nn.LeakyReLU(),
                                    #th.nn.Linear(nh,nh),
                                    th.nn.ReLU(),
                                    th.nn.Linear(nh,1)
                                    )
        self.batch_size = batch_size

    def forward(self, x):
        """Pass data through the net structure.
        :param x: input data: shape (:,1)
        :type x: torch.Variable
        :return: output of the shallow net
        :rtype: torch.Variable
        """
        self.noise.normal_()
        return self.layers(th.cat([x, self.noise], 1))

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class CausalMmdNet(PairwiseModel):
    """ A more complex class that creates 2 shallow networks for X-->Y and Y-->X
        using mmd as a criterion (MmdNet generative model)
        """
    def __init__(self, nh=20, lr=0.01, n_kernels=5, weight_decay=1e-3,
                gpus=None, verbose=None, device=None):
        """Init the model."""
        super(CausalMmdNet, self).__init__()
        self.gpus = SETTINGS.get_default(gpu=gpus)
        self.device = SETTINGS.get_default(device=device)
        self.nh = nh
        self.lr = lr
        self.wd = weight_decay
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.dataloader = []
        self.te_loss_causal = []
        self.te_loss_anticausal = []
        self.batch_size = -1
        self._fcm_net_causal = None
        self._fcm_net_anticausal = None
        self.optim_causal = None
        self.optim_anticausal = None
        self.data_is_set = False ; self.trained = False
        self.te_data_is_set = False


    def set_data(self, data, batch_size = (-1) ):
        """ Given data, creates nets & optimizers, prepare for training."""

        if batch_size == -1:
            # put the whole sample if not specified
            # careful, when using TensorDataset objects, length and shape are different
            self.batch_size = data[0].__len__()
        else:
            self.batch_size = batch_size

        self._fcm_net_causal = MmdNet(self.batch_size, nh=self.nh).to(self.device)
        self._fcm_net_anticausal = MmdNet(self.batch_size, nh=self.nh).to(self.device)
        #print(data.shape)
        self._data = [th.Tensor(scale(th.Tensor(i).view(-1, 1))) for i in data]
        #print([d.shape for d in data])
        self._data = TensorDataset(self._data[0].to(self.device) , self._data[1].to(self.device))
        self.dataloader = DataLoader(   data, batch_size=self.batch_size,
                                        shuffle=True, drop_last=True)

        # reset for every pair
        self._fcm_net_causal.reset_parameters()
        self._fcm_net_anticausal.reset_parameters()

        self.optim_causal = th.optim.Adam(self._fcm_net_causal.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optim_anticausal = th.optim.Adam(self._fcm_net_anticausal.parameters(), lr=self.lr, weight_decay=self.wd)

        self.data_is_set = True

    def fit_two_directions(self, data, train_epochs=1000, idx='no index'):
        ''' fits the two networks using the same number of epochs '''

        self._fcm_net_causal.train() ; self._fcm_net_anticausal.train()

        bar = trange(train_epochs)

        for epoch in pbar:
            for i, (_X, _Y) in enumerate(self.dataloader):
                self.optim_causal.zero_grad()
                self.optim_anticausal.zero_grad()

                # generate fake FCM data
                _Y_hat = self._fcm_net_causal(_X)
                _X_hat = self._fcm_net_anticausal(_Y)

                _XY = th.cat((_X,_Y),1)

                _XY_hat_causal = th.cat( (_X, _Y_hat) ,1)
                _XY_hat_anticausal = th.cat( (_X_hat, _Y) ,1)

                loss_causal = self._fcm_net_causal.criterion(_XY, _XY_hat_causal)
                loss_anticausal = self._fcm_net_anticausal.criterion(_XY, _XY_hat_anticausal)

                loss_causal.backward()
                loss_anticausal.backward()
                self.optim_causal.step()
                self.optim_anticausal.step()

                if not epoch % (int(1+train_epochs/10)) and i == 0:
                    # get all info from NN classes
                    pbar.set_postfix(   idx=idx,
                                        score=( loss_causal.item(),
                                                loss_anticausal.item()
                                                ))

        self.trained = True

    def generate_both_models(self):
        ''' samples ONE point y for each x in the data.
            the data is set using the `set_data` method.
        '''
        assert self.data_is_set

        self._fcm_net_anticausal.eval() ; self._fcm_net_anticausal.eval()

        for i, (_X, _Y) in enumerate(self.dataloader):
            _Y_hat = self._fcm_net_causal(_X)
            _X_hat = self._fcm_net_anticausal(_Y)
            yield (_X, _Y_hat) , (_X_hat, _Y)


    def sample_at_new_point(self, input_value , direction, num_samples=100):
        """ given a new value of x (or y), generates samples `num_samples` samples
            from the corresponding FCM, which is chosen using the `direction` param.
        """

        assert direction in ['->', '<-', 'causal', 'anticausal']
        self._fcm_net_anticausal.eval() ; self._fcm_net_anticausal.eval()

        noise = np.random.normal(0,1,size=num_samples)
        noise = th.from_numpy(noise).float().view(-1,1).to(self.device)

        # copy it
        input_value = input_value*th.ones(num_samples).float().view(-1,1).to(self.device)

        if direction in ['->','causal']:
            _out = self._fcm_net_causal.layers(th.cat([input_value, noise], 1))
        else:
            _out = self._fcm_net_anticausal.layers(th.cat([input_value, noise], 1))

        return _out.detach().cpu().numpy()

    def penalize_weight(self,p=2):
        _W_norm_causal, _W_norm_anticausal = 0.0 , 0.0
        for param in self._fcm_net_causal.parameters():
            _W_norm_causal += th.norm(param,p=p).detach().cpu().numpy()
        for param in self._fcm_net_anticausal.parameters():
            _W_norm_anticausal += th.norm(param,p=p).detach().cpu().numpy()
        return _W_norm_causal, _W_norm_anticausal

    def mmd_scores(self, test_epochs):
        ''' estimates MMD( XY, XY_hat) by resampling XY_hat `test_epoch` times
            & then averaging.
        '''
        self._fcm_net_anticausal.eval() ; self._fcm_net_anticausal.eval()
        pbar = trange(test_epochs)

        for epoch in pbar:
            # the generator splits each into batches theoretically, in practice full dataset
            for (_X, _Y_hat) , (_X_hat, _Y) in self.generate_both_models():

                _XY = th.cat((_X, _Y), 1)

                _XY_hat_causal = th.cat( (_X, _Y_hat) ,1)
                _XY_hat_anticausal = th.cat( (_X_hat, _Y) ,1)

                loss_causal = self._fcm_net_causal.criterion(_XY, _XY_hat_causal)
                loss_anticausal = self._fcm_net_anticausal.criterion(_XY, _XY_hat_anticausal)

                self.te_loss_causal.append(loss_causal.detach().cpu().numpy())
                self.te_loss_anticausal.append(loss_anticausal.detach().cpu().numpy())

        self.mmd_score_causal = np.array(self.te_loss_causal).mean()
        self.mmd_score_anticausal = np.array(self.te_loss_anticausal).mean()

        return self.mmd_score_causal, self.mmd_score_anticausal

    def generate_conditional_sampling(self, pair, n_cause=1000,
                                            n_out_per_pt=100, sampling_type='unif'):
        """ in each direction, samples `n_out_per_pt` points per input, where
            inputs are taken from `pair`=: [_X, _Y] a np.array of dims [N,2].
            if the input is X, samples at each _x in _X  `n_out_per_pt` points
            from the causal fcm  .
            if  `sampling_type` is not 'sample', replaces 'pair' by a synthetic input
        """
        assert sampling_type in ['unif', 'sample', 'mixtunif']

        self.te_X, self.te_Y = cause_sample(pair, sampling_type, n_cause=n_cause)

        te_uX, te_uY = np.unique(self.te_X), np.unique(self.te_Y)

        self.te_Y_hat = [ self.sample_at_new_point(_x, 'causal', num_samples=n_out_per_pt) for _x in te_uX ]
        self.te_X_hat = [ self.sample_at_new_point(_y, 'anticausal', num_samples=n_out_per_pt) for _y in te_uY ]

        self.te_data_is_set = True

    def estimate_conditional_var(self):
        """ in each direction, computes the conditionsal var[y|x]
            if test values have already been generated using `generate_conditional_sampling`
        """
        assert self.te_data_is_set

        te_var_causal = [ _y_hat.std()**2 for _y_hat in self.te_Y_hat ]
        te_var_anticausal = [ _x_hat.std()**2 for _x_hat in self.te_X_hat ]

        self.te_var_causal = np.array(te_var_causal)
        self.te_var_anticausal = np.array(te_var_anticausal)


    def estimate_conditional_quants(self, probs=None):
        """ in each direction, computes the conditionsal quantiles for _out | _in ,
            if test values have already been generated using `generate_conditional_sampling`
        """
        assert self.te_data_is_set

        probs = probs if (probs is not None) else np.arange(1,10)/10.0

        self.te_quants_causal = [ hdquantiles(_y_hat, prob=probs) for _y_hat in self.te_Y_hat ]
        self.te_quants_anticausal = [ hdquantiles(_x_hat, prob=probs) for _x_hat in self.te_X_hat ]

        self.te_qscores_causal = [  itg.simps(  qscore(self.te_quants_causal[i], probs, _y_hat),
                                                probs)
                                    for i, _y_hat in enumerate(self.te_Y_hat) ]

        self.te_qscores_anticausal = [  itg.simps(  qscore(self.te_quants_anticausal[i], probs, _x_hat),
                                                    probs)
                                    for i, _x_hat in enumerate(self.te_X_hat) ]

        self.te_qscores_causal = np.array(self.te_qscores_causal)
        self.te_qscores_anticausal = np.array(self.te_qscores_anticausal)

    def compute_variability(self, stat_name, variability_type):
        assert stat_name in ['quantiles', 'variances']
        assert variability_type in ['mean', 'max']

        if stat_name == 'quantiles':
            assert hasattr(self, 'te_qscores_causal'), hasattr(self, 'te_qscores_anticausal')
            stat_causal = self.te_qscores_causal
            stat_anticausal = self.te_qscores_anticausal

        elif stat_name == 'variances':
            assert hasattr(self, 'te_var_causal'), hasattr(self, 'te_var_anticausal')
            stat_causal = self.te_var_causal
            stat_anticausal = self.te_var_anticausal

        # given stat type fixed, compute diffs & mean
        _diffs_causal = _get_diffs(stat_causal)
        _diffs_anticausal = _get_diffs(stat_anticausal)

        if variability_type == 'mean':
            return _diffs_causal.mean() , _diffs_anticausal.mean()
        elif variability_type == 'max':
            return _diffs_causal.max() , _diffs_anticausal.max()

    def add_penalty(self, stat_name='norm', variability_type=None, beta=0.5 ,p=2):
        # if norm penalty, do not check variability_type
        assert stat_name in ['quantiles', 'variances', 'norm']

        if stat_name == 'norm':
            penalty_causal, penalty_anticausal = self.penalize_weight(p)

        else:
            assert variability_type in ['mean', 'max']
            penalty_causal, penalty_anticausal = self.compute_variability(stat_name, variability_type)

        if not hasattr(self, 'mmd_score_causal') or not hasattr(self, 'mmd_score_anticausal'):
            self.mmd_scores(500) # base 500 te_epochs
        else:
            mmd_causal, mmd_anticausal = self.mmd_score_causal, self.mmd_score_anticausal


        score_causal = mmd_causal + beta*penalty_causal
        score_anticausal = mmd_anticausal + beta*penalty_anticausal

        _total = score_causal+score_anticausal

        score_causal /= _total ; score_anticausal /= _total

        return score_causal, score_causal







    def save_checkpoint(self,filepath,chkpt_epoch,idx=0, nrun=0):
        param_desc = '_'.join((
                            'nh'+str(self.nh),
                            'ep',
                            str(chkpt_epoch),
                            'idx',
                            str(idx),
                            'nrun',
                            str(nrun))
                            )

        modelstr = '_'.join((
                            'nh'+str(self.nh)
                            ))

        root_dir = filepath+"/save_checkpoints/"+modelstr
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        path = root_dir+"/"+param_desc

        th.save({   'fcm_causal':self.FCM_XY.state_dict(),
                    'fcm_anticausal':self.FCM_YX.state_dict(),
                    'nh':self.nh,
                    'wd':self.wd,
                    'epoch':chkpt_epoch,
                    'score_XY':self.mmd_score_causal,
                    'score_YX':self.mmd_score_anticausal,
                    'var_xy':self.te_var_causal,
                    'var_yx':self.te_var_anticausal
                    }, path)
        self.last_checkpoint = chkpt_epoch
        return path





################### Func for different resampling of cause domain ##########

def _get_diffs(vector):
    ''' inputs a [n,] numpy array and computes all the x_i - x_j in a matrix
    '''
    _diffs = np.abs( vector.reshape(-1,1) - vector.reshape(-1,1).T )
    _diffs = _diffs[ np.triu_indices_from(_diffs, k=1) ]

    return _diffs

def cause_sample( pair, sampling_type, n_cause=1000):

    _X_std, _Y_std = pair[0].std(), pair[1].std()

    if sampling_type == 'sample':
        _X_syn, _Y_syn = pair[0], pair[1]

    elif sampling_type == 'unif':
        n_cause = min(n_cause, 700) # can be too expensive
        # sample from the interior of the support
        # might crash if too few & concentrated _x values
        _X_lb, _X_ub  = min(pair[0]) + _X_std*0.01, max(pair[0]) - _X_std*0.01
        _X_lb, _X_ub = min(_X_lb, _X_ub), max(_X_lb, _X_ub)

        _X_syn = np.linspace(_X_lb,_X_ub, n_cause)

        _Y_lb, _Y_ub  = min(pair[1]) + _Y_std*0.01, max(pair[1]) - _Y_std*0.01
        _Y_lb, _Y_ub = min(_Y_lb, _Y_ub), max(_Y_lb, _Y_ub)

        _Y_syn = np.linspace(_Y_lb,_Y_ub, n_cause)

    elif typecause=='mixtunif':

        _X_shift = (max(pair[0])- min(pair[0]))*1e-2 # 1% of the range
        _Y_shift = (max(pair[1])- min(pair[1]))*1e-2

        samples_per_point = n_cause // len(pair[0])

        if samples_per_point:
            # generate samples_per_point new pts per _X value
            _X_disps = np.linspace( -_X_shift, +_X_shift, samples_per_point)
            _Y_disps = np.linspace( -_Y_shift, +_Y_shift, samples_per_point)

            _X_syn = np.concatenate( [ _x + _X_disps for _x in pair[0] ] )
            _Y_syn = np.concatenate( [ _y + _Y_disps for _y in pair[1] ] )

        else:
            # only displace 3 times per point; ideally should subsample the _X's and _Y's ...
            _X_disps = np.array( [-shift_x , 0 , shift_x] )
            _Y_disps = np.array( [-shift_y , 0 , shift_y] )

            _X_syn = np.concatenate( [ _x + _X_disps for _x in pair[0] ] )
            _Y_syn = np.concatenate( [ _y + _Y_disps for _y in pair[1] ] )

    return _X_syn, _Y_syn


def qscore(quantiles, probs, y_sample):
        ''' given a quantile of y|x and a sample y from y|x,
            compute the quantile score, assumes y_sample is [n,1]
            and that the quantiles match with their probs,
            namely quantiles[i] = hdquantile(probs[i])
        '''
        #print(quantiles.shape, y_sample.shape)
        #qs = np.array([q-y for y in y_sample if y<= q])
        _quants = quantiles.reshape(-1,1)
        _diffs = _quants - y_sample.T
        _qs = np.where(_diffs >=0, _diffs, 0)

        #print(quants.shape, diffs.shape, qs.shape)
        # also need the  - probs * (q - condexp)
        diff_mean = - probs.reshape(-1,1) * (_quants - y_sample.mean(0)*np.ones(_quants.shape))

        # each element is q_i - y_j  , and we only keep positive elements,
        # corresponding to the constraint: fixing q_i, only sum those y s.t q_i >= y
        # then sum up the row over all cols, corresponding to the average quantile residuals
        return qs.mean(1) + diff_mean.ravel()
