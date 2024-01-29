import os
import numpy as np
from itertools import product

def load_data(dataset, data_dir_path):
    L_train, L_test, y_test = [], [], []

    for g in ['train', 'dev', 'test']: 
        if os.path.exists(os.path.join(data_dir_path, '%s_L_%s.npy' % (dataset, g))):
            L = np.load(os.path.join(data_dir_path, '%s_L_%s.npy' % (dataset, g)))
            L_train.append(L)

            if os.path.exists(os.path.join(data_dir_path, '%s_y_%s.npy' % (dataset, g))):
                y = np.load(os.path.join(data_dir_path, '%s_y_%s.npy' % (dataset, g)))
                L_test.append(L)
                y_test.append(y)

    L_train = np.concatenate(L_train)
    L_test = np.concatenate(L_test)
    y_test = np.concatenate(y_test)
    return L_train, L_test, y_test

if __name__ == '__main__':
    L_train, L_test, y_test = load_data('rr_hi')
    print('Train size', L_train.shape)
    print('Test size', L_test.shape)
    assert(L_test.shape[0] == y_test.shape[0])