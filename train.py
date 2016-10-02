# path to training images & validation images
# batch size
# gpu_id (-1 for cpu)
# save option
# chainer training loop (cpu loading)


from patches import *
from jigsaw import Jigsaw
from alexnet import AlexNet
from permutations import permute_patches, load_permutation_indices

import numpy as np

import chainer

from chainer import optimizers

import collections

import glob

from PIL import Image


permutation_indices = load_permutation_indices()

anet = AlexNet()
anet.to_gpu(1)
jigsaw = Jigsaw(anet)
jigsaw.to_gpu(1)

losses = collections.deque(maxlen=20)
accs = collections.deque(maxlen=20)

optimizer = optimizers.MomentumSGD(lr=0.01)
optimizer.setup(jigsaw)

batch_x = []
batch_y = []
b = 0

for image in glob.glob('/data/eddie/multi-image/images/*/*.jpg'):
    image = Image.open(image)
    image_array = np.asarray(image)

    if len(image_array.shape) != 3:
        continue

    image_array = image_array.transpose(2,1,0)

    crop = random_crop(image_array)
    patches = random_patches(crop)
    patches = np.concatenate(patches).reshape(9, 3, 64, 64)

    label, permuted_patches = permute_patches(permutation_indices, patches)

    batch_y.append(label)
    batch_x.append(permuted_patches)

    if len(batch_y) == 192:
        b+=1
        # patch dimension first
        batch_x = np.array(batch_x).astype(np.float32)
        batch_x = batch_x.transpose(1,0,2,3,4)

        batch_y = np.array(batch_y).astype(np.int32)
        batch_y.reshape(1, -1)

        batch_x = chainer.Variable(batch_x)
        batch_y = chainer.Variable(batch_y)

        batch_x.to_gpu(1)
        batch_y.to_gpu(1)

        loss, acc = jigsaw(batch_x, batch_y)

        optimizer.zero_grads()
        loss.backward()
        optimizer.update()

        loss.to_cpu()
        losses.append(loss.data)
        acc.to_cpu()
        accs.append(acc.data)

        print '%d %.5f (%.5f)' % (
            b,
            sum(losses) / len(losses),
            sum(accs) / len(accs),
        )

        batch_x = []
        batch_y = []

