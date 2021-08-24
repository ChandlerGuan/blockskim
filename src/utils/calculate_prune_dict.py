import numpy as np


def calculate_prune_dict(file_name='tmp/head_pruning/prune_single_head.csv', k=6):
    acc = np.loadtxt(open(file_name,'rb'), delimiter=',')
    idx = np.argsort(acc, axis=1)[:,:k]
    return_dict = {layer_idx:idx[layer_idx].tolist() for layer_idx in range(idx.shape[0])}
    return return_dict

if __name__ == "__main__":
    print(calculate_prune_dict())

