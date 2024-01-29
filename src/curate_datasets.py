import argparse
import os
import networkx as nx
import numpy as np
import statsmodels.stats.proportion as sm
from math import sqrt
from scipy.stats import spearmanr
# Probabilistic Labelers
# from majorityvote import MajorityVote
from snorkel.labeling.model import LabelModel as GenerativeModel

parser = argparse.ArgumentParser()
parser.add_argument('--pathToData', type=str, help='path to the data files L_{...}.npy')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--corThresh', type=float, default=0.5, help='LF correlation threshold')
parser.add_argument('--probLabeler', type=str, default='gm', help='probabilistic labeler, i.e., \"mv\" for majority vote or \"gm\" for generative model')
parser.add_argument('--alpha', type=float, default=0.05, help='confidence interval significance level')
parser.add_argument('--nDatasets', type=int, default=10, help='number of datasets to curate')
parser.add_argument('--eval', type=bool, default=True, help='flag to evaluate dataset accuracies')
parser.add_argument('--defaultClass', type=int, default=0, help='default class, i.e., what class to use in event of abstain')
parser.add_argument('--zStdDev', type=bool, default=1.64, help='number of standard deviations for accuracy confidence intervals')

ABSTAIN     = -1
N_CLASS     = 2
CI_METHOD   = 'beta'

class MajorityVote:
    def __init__(self, n_class, abstain):
        self.n_class = n_class
        self.abstain = abstain

    def __get_logits__(self, l):
        return np.bincount(l[l != self.abstain], minlength=self.n_class)
        
    def predict_proba(self, L, method="softmax"):
        logits = np.apply_along_axis(self.__get_logits__, 1, L)
        
        if method == "softmax":
            return softmax(logits, 1)
        elif method == "total votes":
            return logits / np.sum(logits, 1)[:,None]

binary_conversion = {-1: 0, 0: -1, 1: 1}
convert_to_binary = np.vectorize(binary_conversion.get)

def abstaining_argmax(X, default=ABSTAIN):
    f = lambda x: np.argmax(x) if np.unique(x).shape[0] > 1 else default
    return np.apply_along_axis(f, 1, X)

def lf_pruning(L, cor_thresh):
    '''
    Select independent labeling functions.

    args:
        L (np.array) : n x m of LF output matrix where n is the number of samples and m is the number of LFs
        cor_thresh (float) : correlation threshold for dependence

    return:
        (tuple) independent LF output matrix (np.array) and indicies of dependent LFs (list<int>)
    '''
    lfs = np.arange(L.shape[1])
    # Compute LF correlations
    cor_mtx = np.corrcoef(L.T)
    cor_mtx = np.tril(cor_mtx, -1)
    cor_mtx = np.abs(cor_mtx)
    # Construct dependency graph
    dep_graph = nx.Graph()
    dep_graph.add_nodes_from(lfs)
    dep_graph.add_edges_from(np.column_stack(np.nonzero(cor_mtx > cor_thresh)))
    # Rank LFs by maximal cliques and coverage
    max_cliques = list(nx.find_cliques(dep_graph))
    lf_cliques_in = {lf: [] for lf in lfs}
    for i, clique in enumerate(max_cliques):
        for lf in clique:
            lf_cliques_in[lf].append(i)
    lf_n_cliques = map(len, lf_cliques_in.values())
    lf_coverages = (L != ABSTAIN).sum(axis=0) / L.shape[0]
    lf_ranking = sorted(zip(lfs, lf_n_cliques, lf_coverages), key=lambda x: (-x[1], -x[2]))
    # Iteratively drop highly correlated LFs
    lfs_to_drop = lfs
    for lf, _, _ in lf_ranking:
        if lf in lfs_to_drop:
            for clique in lf_cliques_in[lf]:
                c = max_cliques[clique]
                c.remove(lf)
                lfs_to_drop = np.setdiff1d(lfs_to_drop, c)
    L = np.delete(L, lfs_to_drop, axis=1)
    return L, lfs_to_drop

def prob_lbling(L_train, L_test, pl, seed, return_weights=False):
    '''
    Probabilistically label the data

    args:
        L_train (np.array) : n x m of LF output matrix where n is the number of samples and m is the number of LFs
        L_test (np.array) : n x m of LF output matrix where n is the number of samples and m is the number of LFs
        pl (str) : probabilistic labeler to use, i.e., 'mv' for majority vote or 'gm' for generative model, default is 'gm'
        seed (int) : random seed
        return_weights (bool) : indicator to return LF weights

    return
        (np.array) n x 2 array of probabilistic labels
    '''
    if pl == 'mv':
        labeler = MajorityVote(cardinality=N_CLASS)
    elif pl == 'gm':
        labeler = GenerativeModel(cardinality=N_CLASS)
        labeler.fit(L_train, seed=seed)
    else:
        print('Probabilistic labeler \"%s\" not supported.' % pl)
        raise NotImplementedError

    y_prob = labeler.predict_proba(L_test)

    if return_weights:
        weights = labeler.get_weights()
        if weights.ndim != N_CLASS:
            weights = np.tile(weights, (N_CLASS, 1))
        return y_prob, weights

    return y_prob

def conf_ints(L, y_prob, lf_weights, alpha):
    '''
    Confidence intervals for probabilistic labels

    args:
        L (np.array) : n x m of LF output matrix where n is the number of samples and m is the number of LFs
        y_prob (np.array) : n x 2 array of probabilistic labels
        lf_weights (np.array) : m-dimensional array of LF weights
        alpha (float) : significance level

    return:
        (np.array) n x 2 array of confidence intervals
    '''
    L = convert_to_binary(L)

    y_ints = np.zeros((L.shape[0], 2)) * np.nan
    for i, (weak_labels, prob_label) in enumerate(zip(L, y_prob)):
        # Calculate the weighted label votes
        weighted_weak_labels = [np.sum(lf_weights[c] * weak_labels) for c in range(N_CLASS)]
        # Calculate voting weights
        voting_weights = np.exp(weighted_weak_labels)
        voting_weights = voting_weights / np.sum(voting_weights)
        # scale to number of total votes (to calculate confidence interval)
        num_voters = np.sum(weak_labels != 0)
        scaled_voting_weights = voting_weights * num_voters
        # compute clopper-pearson confidence interval
        n_successes = int(np.round(scaled_voting_weights[np.argmax(prob_label)]))
        n_trials = num_voters
        if n_trials == 0:
            cil, cih = 0, 1
        else:
            cil, cih = sm.proportion_confint(n_successes, n_trials, alpha, CI_METHOD)
        y_ints[i] = [cil, cih]
        # print(weak_labels, prob_label, voting_weights, num_voters, scaled_voting_weights, y_ints[i])

    assert(~np.all(np.isnan(y_ints)))
    return y_ints

def dataset_design(sample_adv_order, n_datasets):
    '''
    Select samples for adversarially ordered natural datasets

    args:
        sample_adv_order (list) : list of adversarially ordered sample indices
        n_datasets (int) : number of datasets

    return:
        (list) list of lists of sample indices for each dataset
    '''
    top_p = np.linspace(0, 1, num=n_datasets+1)
    n_top_p = np.round(top_p * sample_adv_order.shape[0]).astype(int)[1:]
    n_top_p[-1] = -1        # Let the last dataset contain all samples
    idxs_per_dataset = [sample_adv_order[:n] for n in n_top_p]
    return idxs_per_dataset

def evaluate(y_true, y_pred, idxs_per_dataset, z_std):
    accs = [(y_true[idxs] == y_pred[idxs]).sum() / idxs.shape[0] for idxs in idxs_per_dataset]
    ints = [z_std * sqrt((accs[i] * (1 - accs[i])) / idxs_per_dataset[i].shape[0]) for i in range(len(idxs_per_dataset))]
    dataset_accs = list(zip(accs, ints))
    res = spearmanr(range(len(accs)), accs)
    return dataset_accs, res.statistic, res.pvalue


# def evaluate(L, y, y_prob, idxs_per_dataset, default_class, z_std):
#     '''
#     Evaluate datasets

#     args:
#         L (np.array) : n x m of LF output matrix where n is the number of samples and m is the number of LFs
#         y
#         y_prob
#         idxs_per_dataset
#         default_class

#     return:
#         list of dataset accuracies with interval, spearman's rank coefficient, spearman's rank p-value
#     '''
#     y_pred = abstaining_argmax(y_prob, default=default_class)
#     accs = [(y[idxs] == y_pred[idxs]).sum() / idxs.shape[0] for idxs in idxs_per_dataset]
#     ints = [z_std * sqrt((accs[i] * (1 - accs[i])) / idxs_per_dataset[i].shape[0]) for i in range(len(idxs_per_dataset))]
#     dataset_accs = list(zip(accs, ints))
#     res = spearmanr(range(len(accs)), accs)
#     return dataset_accs, res.statistic, res.pvalue

def main(opt):
    rho, pval = np.nan, np.nan
    ## Step 0: Load the data and apply the LFs
    L_trn, L_dev, L_tst = None, None, None
    y_trn, y_dev, y_tst = None, None, None
    if os.path.exists(opt.pathToData + 'L_train.npy'):
        L_trn = np.load(opt.pathToData + 'L_train.npy')
    if os.path.exists(opt.pathToData + 'Y_train.npy'):
        y_trn = np.load(opt.pathToData + 'Y_train.npy')
    if os.path.exists(opt.pathToData + 'L_dev.npy'):
        L_dev = np.load(opt.pathToData + 'L_dev.npy')
    if os.path.exists(opt.pathToData + 'Y_dev.npy'):
        y_dev = np.load(opt.pathToData + 'Y_dev.npy')
    if os.path.exists(opt.pathToData + 'L_test.npy'):
        L_tst = np.load(opt.pathToData + 'L_test.npy')
    if os.path.exists(opt.pathToData + 'Y_test.npy'):
        y_tst = np.load(opt.pathToData + 'Y_test.npy')
    L_train = np.concatenate([L for L in (L_trn, L_dev, L_tst) if L is not None])
    L_test = np.concatenate([L for L, y in zip((L_trn, L_dev, L_tst), (y_trn, y_dev, y_tst)) if y is not None])
    y_test = np.concatenate([y for y in (y_trn, y_dev, y_tst) if y is not None])
    ## Step 1: LF Pruning
    L_train_indep, lfs_to_drop = lf_pruning(L_train, opt.corThresh)
    if L_train.shape[1] - len(lfs_to_drop) < 3:
        if opt.eval:
            print(np.nan, np.nan)
        return None, None
    L_test_indep = np.delete(L_test, lfs_to_drop, axis=1)
    ## Step 2: Probabilistic Labeling
    y_prob, lf_weights = prob_lbling(L_train_indep, L_test_indep, opt.probLabeler, opt.seed, return_weights=True)
    ## Step 3: Confidence Intervals
    y_ints = conf_ints(L_test_indep, y_prob, lf_weights, opt.alpha)
    ## Step 4: Adversarial Dataset Design
    order_by = y_ints[:,0]                  # Order by confidence interval lower bound   
    # order_by = np.max(y_prob, axis=1)       # To run comparative approach, uncomment this line and comment the above line
    sample_adv_order = np.argsort(order_by) 
    idxs_per_dataset = dataset_design(sample_adv_order, opt.nDatasets)
    ## Evaluation
    if opt.eval:
        dataset_accs, rho, pval = evaluate(L_test_indep, y_test, y_prob, idxs_per_dataset, opt.defaultClass, opt.zStdDev)
        # print(dataset_accs, rho, pval)
        print(rho, pval)
    
    return idxs_per_dataset, y_prob    

if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt, flush=True)
    main(opt)