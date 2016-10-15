import numpy as np
import pickle


def load_permutation_indices(path='data/permutations.pkl'):
    return np.array(pickle.load(open(path)))


def permute_patches(indices, patches):
    """ Reorder patches based on a random permutation. The permutations were
        selected such that they are maximally different """
    label = np.random.randint(0, indices.shape[0] - 1)
    permutation = indices[label]
    return label, patches[permutation, :, :, :]
