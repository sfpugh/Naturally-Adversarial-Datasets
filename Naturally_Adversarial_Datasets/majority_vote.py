import numpy as np
from scipy.special import softmax

class MajorityVote:
    def __init__(self, cardinality):
        self.cardinality = cardinality

    def __get_logits__(self, l):
        return np.bincount(l[l != self.abstain], minlength=self.cardinality)
        
    def predict_proba(self, L, method='softmax'):
        logits = np.apply_along_axis(self.__get_logits__, 1, L)
        
        if method == 'softmax':
            return softmax(logits, 1)
        elif method == 'total votes':
            return logits / np.sum(logits, 1)[:,None]
        else:
            raise NotImplementedError('Normalization method \'%s\' not implemented.' % method)