import numpy as np

ABSTAIN = -1

def abstaining_argmax(X, default=ABSTAIN):
    f = lambda x: np.argmax(x) if np.unique(x).shape[0] > 1 else default
    return np.apply_along_axis(f, 1, X)

def convert_to_binary(L):
    '''
    Map label matrix values to binary, i.e., {-1, 0 1} where -1 is an abstain
    to {-1, 0 1} where -1 is negative, 1 is positive, and 0 is abstain
    '''
    binary_conversion = {-1: 0, 0: -1, 1: 1}
    f_convert = np.vectorize(binary_conversion.get)
    return f_convert(L)