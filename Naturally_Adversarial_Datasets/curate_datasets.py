import argparse
import networkx as nx
import numpy as np
import pickle
import statsmodels.stats.proportion as sm
# Probabilistic labeling models
from majority_vote import MajorityVote
from snorkel.labeling.model.label_model import LabelModel as Snorkel

from data import load_data
from utils import convert_to_binary
from evaluate import evaluate, plot

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help='Dataset')
parser.add_argument('--dataset_dir', type=str, default='../data', help='Dataset')

# Method parameters
parser.add_argument('--pl', type=str, help='Probabilistic labeler (\'majorityvote\' or \'snorkel\')')
parser.add_argument('--delta', type=float, default=0.5, help='Correlation threshold alpha')
parser.add_argument('--alpha', type=float, default=0.05, help='Confidence interval significance level alpha')
parser.add_argument('--ascending', action='store_true', help='Flag to save dataset indexes')
parser.add_argument('--N', type=int, default=10, help='Number of datasets')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

parser.add_argument('--skip_lf_pruning', action='store_true', help='Flag to disable lf pruning step')
parser.add_argument('--skip_ci', action='store_true', help='Flag to disable confidence interval step')

parser.add_argument('--save', action='store_true', help='Flag to save dataset indexes')
parser.add_argument('--savefile', type=str, default='dataset_indexes.pkl', help='Filename to save dataset indexes')
parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate')
parser.add_argument('--z', type=float, default=1.64, help='Number of standard deviations for evaluation accuracy confidence intervals')
parser.add_argument('--plot', action='store_true', help='Flag to plot')

ABSTAIN         = -1
CARDINALITY     = 2

def lf_pruning(L, delta):
    '''
    (Step 1) Select independent labeling functions.

    args:
        L (ndarray) : an [n,p] matrix with values in  {-1,0,1,…,k-1}
        delta (float) : correlation threshold

    return:
        lfs_to_drop (list) : list of dependent LFs indicies
    '''
    lfs = np.arange(L.shape[1])
    
    # Compute LF correlations
    cor_mtx = np.corrcoef(L.T)
    cor_mtx = np.tril(cor_mtx, -1)
    cor_mtx = np.abs(cor_mtx)
    
    # Construct dependency graph
    dep_graph = nx.Graph()
    dep_graph.add_nodes_from(lfs)
    dep_graph.add_edges_from(np.column_stack(np.nonzero(cor_mtx > delta)))
    
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
                
    return lfs_to_drop

def probabilistic_labeling(labeler, L_train, L_test, cardinality, return_weights=False, seed=None):
    '''
    (Step 2) Probabilistically label the test data.

    args:
        labeler (str) : probabilistic labeler (i.e., 'snorkel' or 'majorityvote')
        L_train (ndarray) : an [n,p] matrix with values in  {-1,0,1,…,k-1}
        L_test (ndarray) : an [m,p] matrix with values in  {-1,0,1,…,k-1}
        cardinality (int) : cardinality of the data
        return_weights (bool) : flag to return label model weights
        seed (int) : random seed

    return
        y_prob (ndarray) : an [m,k] matrix of probabilistic labels for L_test
        weights (ndarray) : an [p,k] matrix of LF weights per class
    '''

    if labeler == 'majorityvote':
        model = MajorityVote(cardinality=cardinality)
        lf_weights = np.ones((cardinality, L_train.shape[1]))
    elif labeler == 'snorkel':
        model = Snorkel(cardinality=cardinality)
        model.fit(L_train, seed=seed)
        lf_weights = model.get_weights()
        lf_weights = np.tile(lf_weights, (cardinality, 1))      # NOTE Snorkel currently only supports same weights per class 
    else:
        raise NotImplementedError('Probabilistic labeler \"%s\" not implemented.' % labeler)

    y_prob = model.predict_proba(L_test)
    
    return y_prob, lf_weights if return_weights else y_prob

def confidence_intervals(L, y_prob, lf_weights, alpha, cardinality, method='beta'):
    '''
    Confidence intervals for probabilistic labels.

    args:
        L (ndarray) : an [n,p] matrix with values in  {-1,0,1,…,k-1}
        y_prob (ndarray) : an [n,k] matrix of probabilistic labels
        lf_weights (ndarray) : an [p,k] matrix of LF weights per class
        alpha (float) : significance level
        cardinality (int) : cardinality of the data
        method (str) : method to use for confidence interval (see statsmodels supported methods)

    return:
        y_ints (ndarray) : an [n,2] matrix of confidence intervals for predicted class
    '''
    L = convert_to_binary(L)

    y_ints = np.zeros((L.shape[0], 2)) * np.nan
    for i, (weak_labels, prob_label) in enumerate(zip(L, y_prob)):
        # Calculate the weighted label votes
        weighted_weak_labels = [np.sum(lf_weights[c] * weak_labels) for c in range(cardinality)]
        
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
            cil, cih = sm.proportion_confint(n_successes, n_trials, alpha, method)
        
        y_ints[i] = [cil, cih]

    return y_ints

def curate_datasets(order_by, ascending, n_datasets):
    '''
    Select samples for adversarially ordered natural datasets.

    args:
        order_by (ndarray) : an n-dimensional array of CI lower bounds or label confidences
        ascending (bool) : flag for sort in ascending order
        n_datasets (int) : number of datasets

    return:
        dataset_idxs (list) : list of lists containing sample indicies for each adversarial datasets
    '''
    # adversarially order the samples
    adversarial_order = np.argsort(order_by if ascending else -1 * order_by)
    # take the top-p sample subsets from the ordering
    top_p = np.linspace(0, 1, num=n_datasets+1)
    n_top_p = np.round(top_p * adversarial_order.shape[0]).astype(int)[1:]
    n_top_p[-1] = -1        # let the last dataset contain all samples
    idxs_per_dataset = [adversarial_order[:n] for n in n_top_p]
    return idxs_per_dataset

def main(args):
    L_train, L_test, y_test = load_data(args.dataset, args.dataset_dir)

    # Step 1: LF pruning
    if not args.skip_lf_pruning:
        lfs_to_drop = lf_pruning(L_train, args.delta)
        L_train = np.delete(L_train, lfs_to_drop, axis=1)
        L_test = np.delete(L_test, lfs_to_drop, axis=1)

    # Step 2: Probabilistic labeling
    y_prob, lf_weights = probabilistic_labeling(args.pl, L_train, L_test, CARDINALITY, return_weights=True, seed=args.seed)

    # Step 3: Confidence intervals
    if args.skip_ci:
        order_by = np.max(y_prob, axis=1)
    else:
        y_ints = confidence_intervals(L_test, y_prob, lf_weights, args.alpha, CARDINALITY)
        order_by = y_ints[:,0]

    # Step 4: Adversarial dataset design
    idxs_per_dataset = curate_datasets(order_by, args.ascending, args.N)

    if args.save:
        with open(args.savefile, "wb") as fp:
            pickle.dump(idxs_per_dataset, fp)

    if args.evaluate:
        accs, ints, rho, pvalue = evaluate(idxs_per_dataset, y_test, y_prob, args.z)
        print("%.3f (%.2f)" % (rho, pvalue))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)