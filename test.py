from patches import *
from jigsaw import Jigsaw
from alexnet import AlexNet

import numpy as np

img = np.random.uniform(size=(3, 256, 343)).astype(np.float32)
crop = random_crop(img)
patches = random_patches(crop)

batch = np.concatenate(patches).reshape(9, 1, 3, 64, 64)

jigsaw = Jigsaw(AlexNet())
jigsaw(batch, np.array([1], dtype=np.int32))

