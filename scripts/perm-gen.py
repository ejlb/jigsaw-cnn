""" Script for generating permutation set with high average pairwise hamming distance """

import math
import pickle

from itertools import permutations

import numpy as np

from scipy.spatial.distance import hamming

N_PERMS = 100
PERM_LENGTH = 9

perms = []
distances = []

total = float(math.factorial(PERM_LENGTH))

for i, perm in enumerate(permutations(range(PERM_LENGTH))):
    if len(perms) < N_PERMS:
        perms.append(perm)
        continue

    if not distances:
        for pi in perms:
            distances.append(min([hamming(pi, pj) for pj in perms]))

    past = np.min(distances)
    current = np.min([hamming(pi, perm) for pi in perms])

    if current > past:
        idx = np.argmin(np.array(distances))
        del distances[idx]
        del perms[idx]
        perms.append(perm)
        distances.append(current)

        print '%.3f: avg = %.3f' % (i / total, sum(distances) / N_PERMS)

pickle.dump(perms, open('permutations.pkl', 'w'))

perm_distance = []

for pi in perms:
    for pj in perms:
        if pi != pj:
            perm_distance.append(hamming(pi, pj))

print 'average distance: %.5f' % np.mean(perm_distance)
