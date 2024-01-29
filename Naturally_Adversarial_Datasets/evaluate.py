import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.stats import spearmanr

def plot(accs, ints, ax=None, label=None, savefile=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(np.arange(1, len(accs)+1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylim([0, 1])
        ax.grid()
        ax.legend(loc='lower right')
        
    ax.errorbar(np.arange(1, len(accs)+1), accs, ints, label=None)

    if savefile is not None:
        fig.savefig(savefile)

    return fig, ax

def evaluate(dataset_idxs, y_true, y_prob, Z):
    y_pred = (np.max(y_prob, 1) >= 0.5).astype(int)

    accs = [(y_true[idxs] == y_pred[idxs]).sum() / idxs.shape[0] for idxs in dataset_idxs]
    ints = [Z * sqrt((accs[i] * (1 - accs[i])) / dataset_idxs[i].shape[0]) for i in range(len(dataset_idxs))]

    res = spearmanr(range(len(accs)), accs)

    return accs, ints, res.statistic, res.pvalue

