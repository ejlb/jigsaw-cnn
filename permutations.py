import numpy as np
import pickle


def load_permutation_indices(path='data/permutations.pkl'):
    return np.array(pickle.load(open(path)))


def permute_patches(indices, patches):
    """ Reorder patches based on a random permutation. The permutations were
        selected such that they are maximally different """
    label = np.random.randint(0, indices.shape[0] - 1)
    label = np.array([label], dtype=np.int32)
    permutation = indices[label]
    permuted_patches = patches[permutation]

    return permuted_patches, label
