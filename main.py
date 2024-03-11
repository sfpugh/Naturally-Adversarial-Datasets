import argparse
import numpy as np
import os
import pickle

from Naturally_Adversarial_Datasets.data import load_data
from Naturally_Adversarial_Datasets.curate_datasets import lf_pruning, probabilistic_labeling, confidence_intervals, curate_datasets
from Naturally_Adversarial_Datasets.evaluate import evaluate, plot

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help='Dataset')
parser.add_argument('--dataset_dir', type=str, default='./data', help='Dataset')
parser.add_argument('--dataset_cardinality', type=int, default=2, help='Dataset cardinality (i.e., number of classes)')
parser.add_argument('--default_pred', type=int, help='Prediction to make when probabilistic labeler abstains')

# Method parameters
parser.add_argument('--pl', type=str, help='Probabilistic labeler (\'majorityvote\' or \'snorkel\')')
parser.add_argument('--delta', type=float, default=0.5, help='Correlation threshold alpha')
parser.add_argument('--alpha', type=float, default=0.05, help='Confidence interval significance level alpha')
parser.add_argument('--ascending', action='store_true', help='Flag to save dataset indexes')
parser.add_argument('--N', type=int, default=10, help='Number of datasets')
parser.add_argument('--seed', type=int, default=737, help='Random seed')

parser.add_argument('--skip_lf_pruning', action='store_true', help='Flag to disable lf pruning step')
parser.add_argument('--skip_ci', action='store_true', help='Flag to disable confidence interval step')

parser.add_argument('--save', action='store_true', help='Flag to save dataset indexes')
parser.add_argument('--idxs_file', type=str, default='dataset_indexes.pkl', help='Filename to save dataset indexes')
parser.add_argument('--y_prob_file', type=str, default='y_prob.npy', help='Filename to save dataset indexes')
parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate')
parser.add_argument('--z', type=float, default=1.64, help='Number of standard deviations for evaluation accuracy confidence intervals')
parser.add_argument('--plot', action='store_true', help='Flag to plot')

if __name__ == '__main__':
    args = parser.parse_args()

    L_train, L_test, y_test = load_data(args.dataset, args.dataset_dir)

    # Step 1: LF pruning
    if not args.skip_lf_pruning:
        indep_lfs = lf_pruning(L_train, args.delta)
        L_train = L_train[:,indep_lfs]
        L_test = L_test[:,indep_lfs]

    # Step 2: Probabilistic labeling
    y_prob, lf_weights = probabilistic_labeling(args.pl, L_train, L_test, args.dataset_cardinality, return_weights=True, seed=args.seed)

    # Step 3: Confidence intervals
    if args.skip_ci:
        order_by = np.max(y_prob, axis=1)
    else:
        y_ints = confidence_intervals(L_test, y_prob, lf_weights, args.alpha, args.dataset_cardinality)
        order_by = y_ints[:,0]

    # Step 4: Adversarial dataset design
    idxs_per_dataset = curate_datasets(order_by, args.ascending, args.N)

    if args.save:
        # Ensure directory exists
        os.makedirs(os.path.dirname(args.idxs_file), exist_ok=True)

        with open(args.idxs_file, "wb") as fp:
            pickle.dump(idxs_per_dataset, fp)
        
        np.save(args.y_prob_file, y_prob)

    if args.evaluate:
        accs, ints, rho, pvalue = evaluate(idxs_per_dataset, y_test, y_prob, args.z, args.default_pred)
        print("%.3f (%.3f)" % (rho, pvalue))