import argparse
import numpy as np
import pickle

from Naturally_Adversarial_Datasets.data import load_data
from Naturally_Adversarial_Datasets.curate_datasets import lf_pruning, probabilistic_labeling, confidence_intervals, curate_datasets
from Naturally_Adversarial_Datasets.evaluate import evaluate, plot

parser = argparse.ArgumentParser()

# (1) Labeling function pruning args
parser.add_argument('--skip_lf_pruning', action='store_true', help='Flag to skip labeling function pruning step')
parser.add_argument('--lf_correlation_thresh', type=float, default=0.5, help='Labeling function correlation threshold (i.e., delta)')
# (2) Probabilistic labeling args
parser.add_argument('--probabilistic_labeler', type=str, default='snorkel', help='Probabilistic labeler (i.e., \'snorkel\' or \'majorityvote\')')
# (3) Confidence intervals for weak labels args
parser.add_argument('--skip_conf_ints', action='store_true', help='Flag to skip confidence interval step (i.e., use label confidences instead)')
parser.add_argument('--alpha', type=float, default=0.05, help='Confidence interval significance level')
# (4) Adversarial dataset curation args
parser.add_argument('--ascending', action='store_true', help='Flag for direction of adversarial ordering')
parser.add_argument('--N', type=int, default=10, help='Number of datasets')
# Evaluation args
parser.add_argument('--evaluate', action='store_true', help='Evaluation flag')
parser.add_argument('--Z', type=float, default=1.64, help='Number of standard deviations for dataset accuracy confidence intervals')
# Other args
parser.add_argument('--dataset', type=str, help='Dataset')
parser.add_argument('--data_dir_path', type=str, default='./data', help='Path to data directory')
parser.add_argument('--cardinality', type=int, default=2, help='Cardinality of data')
parser.add_argument('--seed', type=int, default=1234567, help='Random seed')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args, flush=True)

    # Load the data
    L_train, L_test, y_test = load_data(args.dataset, args.data_dir_path)

    # (1) Labeling function pruning
    if not args.skip_lf_pruning:
        dep_lfs = lf_pruning(L_train, args.lf_correlation_thresh)
        L_train = np.delete(L_train, dep_lfs, axis=1)
        L_test = np.delete(L_test, dep_lfs, axis=1)

    # (2) Probabilistic labeling
    y_test_prob, lf_weights = probabilistic_labeling(args.probabilistic_labeler, L_train, L_test, cardinality=args.cardinality, return_weights=True, seed=args.seed)

    # (3) Confidence intervals
    if not args.skip_conf_ints:
        y_test_ints = confidence_intervals(L_test, y_test_prob, lf_weights, args.alpha, args.cardinality)

    # (4) Adversarial dataset design
    dataset_idxs = curate_datasets(
        order_by=y_test_ints[:,0] if not args.skip_conf_ints else np.max(y_test_prob, 1),
        ascending=args.ascending, n_datasets=args.N)

    with open('adversarial_datasets_idxs.pkl', 'wb') as fp:
        pickle.dump(dataset_idxs, fp)

    if args.evaluate:
        accs, ints, rho, pval = evaluate(dataset_idxs, y_test, y_test_prob, args.Z)
        plot(accs, ints, savefile='temp.pdf')
        print('Spearman\'s Rho: %.2f (p-value %.2f)' % (rho, pval))
