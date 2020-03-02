import numpy as np
import pandas as pd
import h5py
from scipy.optimize import linprog
import os

# for computational load, cut the max number of pairs in TCEP
def cut_num_pairs(data, num_max=1000,shuffle=False):
    """ Constrains the maximum sample size of the dataset to num_max;
        for random subsampling, we rely on initial permutation of the data;
        if less then threshold (too few datapoints), we add extra samples as
        (X,Y) = (X+N,Y+N) with iid noise and 0 mean,
        based on assumption that some FCM exists Y = f(X,E) it should have some robustness to
        negligible perturbations.
        """
    for idx, pair in data.iterrows():
        n_pair = pair[0].shape[0]
        if n_pair > num_max:
            if shuffle:
                prm = np.random.permutation(n_pair)
                pair[0] , pair[1] = pair[0][prm][:num_max] , pair[1][prm][:num_max]
            else:
                 pair[0] , pair[1] = pair[0][:num_max] , pair[1][:num_max]
        # else:
        #   # need to upsample m points to make it a total of num_max
        #   n = pair[0].shape[0]
        #   s_X , s_Y = min(np.std(pair[0])*1e-3, 1) , min(np.std(pair[1])*1e-3, 1)
        #   m = num_max - n
        #   n_copy = m//n
        #   remain = m - n_copy*n
        #   cpX = [np.copy(pair[0]) for _ in range(n_copy)]
        #   cpY = [np.copy(pair[1]) for _ in range(n_copy)]
        #   cpX += [np.copy(pair[0][:remain])]
        #   cpY += [np.copy(pair[1][:remain])]
        #   upsample_X = np.concatenate(cpX,axis=0)
        #   upsample_Y = np.concatenate(cpY,axis=0)
        #   #print(m,upsample_X.shape, upsample_Y.shape, n)

        #   E_X , E_Y = np.random.laplace(0,s_X,m) , np.random.laplace(0,s_Y,m)
        #   pair[0] = np.concatenate((pair[0] , upsample_X+E_X),axis=0)
        #   pair[1] = np.concatenate((pair[1] , upsample_Y+E_Y),axis=0)



def ensemble_score(all_algos_scores):
    """ given a list of p score matrices for TCEP, shape (n_pairs,2),
        output the ensemble score, as the avg for each pair.
    """
    n_pairs = all_algos_scores[0].shape[0]
    mean_scores = [np.mean([mt[i,:] for mt in all_algos_scores],0) for i in range(n_pairs)]

    return mean_scores


def _get_wd(sample_size):
    """ sets high regularization for low sample size """
    if sample_size < 200:
        return 1e-1
    elif sample_size < 500:
        return 1e-2
    elif sample_size < 1000:
        return 1e-3
    else:
        return 1e-4


def _get_nc(sample_size):
    if sample_size < 500:
        return 1000
    elif sample_size < 1000:
        return 800
    else:
        return 500

def _acc_vs_thresh(scores,labels,test, weighted=True):
      """ the test has to be be a binary thresholded test,
      of the form: f(x) = 1 (if x>T)
                          0 (if -T<x<T)
                         -1 (if -T<x)
      test is a callback such that label_preds = f(scores,thresh)"""
      THRESH = np.arange(0,1,0.01)
      labels = np.array(labels).ravel()

      # we compute for each thresh the acc & decision rate
      res = []
      for t in THRESH:
            preds = np.array(test(scores,t)).ravel()
            decision_index = [i for i,pred in enumerate(preds) if not pred==0]
            dec_rate = len(decision_index) / float(len(preds))
            compare = (preds == labels)[decision_index]
            if weighted:
                weights = TCEP_WEIGHTS[decision_index]
                compare = compare*weights
                normalize = sum(weights)
            else:
                normalize = len(compare)
            acc = sum(compare)/max(1,normalize)
            res.append([dec_rate, acc])

      return res


def _all_predict(scores, threshold=0):
    """ uses thresholding on pairs of similiarity scores [S1,S2]
        to make a decision. The scores are positive, and
        rescaled by (S1+S2). The lower a score, the better.
        Is used whenever multiple tests are run in parallel,
        and the thresholding has to be made for all tests.

        the decision is whether (S2-S1)/(S1+S2) > T or < -T
        """
    preds = []
    for S1,S2 in scores:
        score = (S2 - S1)/(S2 + S1)
        if score >= threshold:
            preds.append([1])
        elif score <= -threshold:
            preds.append([-1])
        else:
            preds.append([0])
    return preds

def acc_v_thresh_wrapper(scores,labels, weighted=True):
    """wrapps _acc_vs_thresh to output (confidence,accuracy) pairs with
        both confidence > 0, and sorted by increasing confidence"""
    tot_acc_v_cfd_med = _acc_vs_thresh(scores,labels,test=_all_predict,
                                      weighted=weighted)
    tot_acc_v_cfd_med = np.array(tot_acc_v_cfd_med).reshape(-1,2)
    cfd = tot_acc_v_cfd_med[:,0].ravel()
    accs = tot_acc_v_cfd_med[:,1].ravel()
    accs, cfd = accs[::-1] , cfd[::-1]
    b = np.greater(accs, 0)
    return cfd[b], accs[b]


## in case the scores are already combined

def thresh_test(s,t):
    if s>t:
        return 1.0
    elif s< -t:
        return -1.0
    else:
        return 0

def thresh_preds(preds,labels, weighted=True):
    THRESH = np.arange(0,1,0.01)
    labels = np.array(labels).reshape(-1,1)
    res = []
    for t in THRESH:
        thr_preds = np.array([thresh_test(p,t) for p in preds]).reshape(-1,1)
        decision_index = [i for i,pred in enumerate(thr_preds) if not pred==0]
        dec_rate = len(decision_index) / float(len(thr_preds))
        compare = (thr_preds == labels)[decision_index].ravel()
        if weighted:
            weights = TCEP_WEIGHTS[decision_index]
            compare = compare*weights
            normalize = sum(weights)
        else:
            normalize = len(compare)
        acc = sum(compare)/max(1,normalize)
        res.append([dec_rate, acc])
    res = np.array(res)
    return res[:,0], res[:,1]

## scores combination methods

def combine(x,y,eps):
    return eps*x + (1-eps)*y
def score_mix(scores1, scores2, eps):
    score1 = combine(scores1[0],scores2[0],eps)
    score2 = combine(scores1[1],scores2[1],eps)
    return [score1, score2]
def mix_all_scores(scores1_list, scores2_list, eps):
    all_scores = []
    for scores1,scores2 in zip(scores1_list,scores2_list):
        all_scores.append(score_mix(scores1, scores2, eps))
    return all_scores
def normalize_scores(scores):
    return np.array([row/sum(row) for row in scores])
def check_nan(scores):
    # in doubt, numerical errors are
    # given an ambiguous score, to reflect
    # the non-usefulness of the score
    scores[np.isnan(scores)] = 0.5
    scores[np.isinf(scores)] = 0.5
    return scores
def scores_to_sep(scores):
    return np.abs(scores[:,1] - scores[:,0])


# methods for LP combination of scores

def reweight_scores(allseps, scores, lb=0, printcoeffs=False):
    # use small scale LP to get coeffs maximizing confidence
    coeffs = get_coeffs(allseps,lb)
    if printcoeffs:
        for i,c in enumerate(coeffs):
            print('coeff for pair #',i)
            print('coeffs:',c)
            print('argmax:',np.argmax(c))
    # get reweighted preds from old scores
    preds_nothresh = np.hstack([
                            s[:,1].reshape(-1,1) for s in scores])\
                            - np.hstack([s[:,0].reshape(-1,1) for s in scores])
    weighted_preds = (coeffs * preds_nothresh).sum(1)
    return weighted_preds

def get_coeffs(allseps,lb):
    score_coeffs = []
    if len(allseps[0])*lb > 1:
        # if lower bound too high, take it as
        # a proportion of the uniform lower bound 1/len
        lb = lb/len(allseps[0])
    sum_constraint = np.ones((1,len(allseps[0])))
    for sp in allseps:
        res = linprog(c=-sp,
                      A_eq=sum_constraint, b_eq=1,
                      bounds=(lb,1))
        score_coeffs.append(res.x)
    return np.array(score_coeffs)

# file handling methods

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

def _to_dataframe(dataset):
    """ puts all the synthetic pairs into
    CEP dataframe format, as seen in the CDT
    package"""
    df_synth = pd.DataFrame({
    'A': [row for row in dataset[:,0,:]],
    'B': [row for row in dataset[:,1,:]]
    })

    return df_synth



def find_between(s, start, end):
    if not (start in s and end in s):
        return s
    else:
        return (s.split(start))[1].split(end)[0]

def _build_scoredir(rootdir, crop=None, distsep='<'):
    score_dir = {}
    for f in os.listdir(rootdir):
        # iterate over datatype
        k = f.split('_reg')[0]
        if crop!=None:
            k = k.replace(crop,'')
        if 'qs' in f:
            if 'mean' in f:
                k = k + '_mean'
            elif 'max' in f:
                k = k + '_max'
            else:
                k = k+ '_nodiff'
        if not k in score_dir:
            if not 'unreg' in k:
                score_dir[k]={}
        if 'unreg' in k:
            score_dir['unreg'] = normalize_scores(check_nan(np.load(rootdir+f)))
        elif 'model' in k:
            score_dir['model'] = normalize_scores(check_nan(np.load(rootdir+f)))
        else:
            if distsep=='<':
                dist_t = find_between(f,'<','>')
            elif distsep=='__':
                dist_t = find_between(f,'__','__')
            score_dir[k][dist_t] = normalize_scores(check_nan(np.load(rootdir+f)))

    return score_dir


# code to remove low-density points using rough binning

def cut_low_density(pair, fracbins=50, axis=0):
    N = pair[axis].shape[0]
    # always use N//100 or 5 bins; maybe tune that later
    nbin = max(N//fracbins, 5)
    bincounts, bin_edges = np.histogram(pair[axis], bins=nbin)
    # nudge the last bin a tiny amout above the max value
    bin_edges[-1] = bin_edges[-1] + 1e-06

    # bincount[i] is the counts between bin_edges[i],bin_edges[i+1]
    # we have bin_edges[indexes[i]-1] <= pair[axis][i] < bin_edges[indexes[i]]

    indexes = np.digitize(pair[axis],bin_edges)

    # the index is the upper edge, so start at 1, and end at bincount+1
    lowcount_bins = np.array([i+1 for i,c in enumerate(bincounts) if c<1.5])
    # for each index, check if a low bin index matches
    # row is for data point index, column is for lowcount_bin index
    index_matches_lowcount =  (indexes.reshape(-1,1) == lowcount_bins.reshape(1,-1))
    # we only care that each data value is in any lowcount bin,
    # therefore any along the cols
    index_matches_lowcount = index_matches_lowcount.any(1)
    #only keep highcount data points
    highcount_mask = np.logical_not(index_matches_lowcount)

    newpair = np.vstack(
        [
        pair[0][highcount_mask],
        pair[1][highcount_mask]])

    return newpair

def wraps_cutlowd(pair,direction='->'):
    if direction == '<-':
        pair = cut_low_density(pair, axis=1)
        return pair[0], pair[1]
    elif direction == '->':
        pair = cut_low_density(pair, axis=0)
        return pair[0], pair[1]
    else:
        raise NotImplementedError('direction type', direction)


# after deleting weights 52 53 54 55, 71 with weights 0 or "16 0"

TCEP_WEIGHTS = [0.166,0.166,0.167,0.166,0.143,0.143,
                0.143,0.143,0.143,0.143,0.142,0.5,
                0.25,0.25,0.25,0.25,0.5,1,1,0.166,
                0.167,0.333,0.333,0.334,0.125,0.125,
                0.125,0.125,0.125,0.125,0.125,0.125,
                0.2,0.2,0.2,0.2,0.2,0.25,0.25,0.25,
                0.25,0.5,0.25,0.25,0.25,0.25,1,1,0.333,
                0.333,0.334,0.083,0.083,0.084,0.083,
                0.083,0.084,0.083,0.083,0.084,0.333,
                0.333,0.334,1,1,1,1,0.083,0.083,0.084,
                1,0.5,0.3333,0.3333,0.3334,0.3333,0.3333,
                0.3334,1,1,1,1,1,0.25,0.25,0.25,0.25,1,
                0.3333,0.3333,0.3333,0.2,0.2,1.0,1.0,0.5,
                0.2,0.2,0.2]


TCEP_WEIGHTS = np.array(TCEP_WEIGHTS).astype('float32')
