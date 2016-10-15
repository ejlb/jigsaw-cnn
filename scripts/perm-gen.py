import pickle

from itertools import permutations

import numpy as np

from scipy.spatial.distance import hamming

perms = []
distances = []

for i, perm in enumerate(permutations(range(9))):
    if len(perms) < 100:
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

        print i
        print 'sum = %.3f avg = %.3f' % (sum(distances), sum(distances) / 100)

pickle.dump(perms, open('permutations.pkl', 'w'))
