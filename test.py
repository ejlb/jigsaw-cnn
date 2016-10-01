from patches import *
from jigsaw import Jigsaw
from alexnet import AlexNet
from permutations import permute_patches, load_permutation_indices

import numpy as np

import chainer

permutation_indices = load_permutation_indices()

img = np.random.uniform(size=(3, 256, 343)).astype(np.float32)
crop = random_crop(img)
patches = random_patches(crop)
patches = np.concatenate(patches).reshape(9, 3, 64, 64)

label, permuted_patches = permute_patches(permutation_indices, patches)

batch = chainer.Variable(np.expand_dims(permuted_patches, axis=1))

jigsaw = Jigsaw(AlexNet())
jigsaw(batch, np.array([label], dtype=np.int32))
