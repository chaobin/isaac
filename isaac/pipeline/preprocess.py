import numpy as np


__all__ = [
    "mean_normalizer",
    "get_XY_from_frame"
]


def mean_normalizer(arr):
    '''
    Update the column in-place using mean normalize.
    '''
    for i in range(arr.shape[1]): # column by column
        column = arr[:, i]
        mean_f, max_f = np.mean(column), np.max(column)
        arr[:, i] = (column - mean_f) / max_f # mean normalization

def get_XY_from_frame(arr, columns, outcome=-1,
        normalize='mean', add_bias=True):
    '''
    arr
        np.narray, the training data
    
    columns
        list, list of indexes of input
    
    outcome
        int, the outcome column, defaults to the last column
    
    return
        tuple, (X, Y)
        
    '''
    len_data = arr.shape[0]
    Y = arr[:, outcome]
    X = arr[:, columns]
    normalizer = normalize + "_normalizer"
    globals()[normalizer](X)
    if add_bias:
        X = np.hstack((np.ones((len(Y), 1)), X)) # adding one bias column
    return (X, Y)
