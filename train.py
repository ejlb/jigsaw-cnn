# validation set
# chainer loop / dataset

# number of permutations and generation method
    # look at confusion matrix
# number of patches
# drop patch
# visualisations

# -1 for cpu
# save option
# min-dim & crop_size


from patches import *
from jigsaw import Jigsaw
from permutations import permute_patches, load_permutation_indices

import numpy as np

import chainer
import argparse
from chainer import optimizers

import collections

import glob

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Train jigsaw CNN.')

    parser.add_argument('--batch-size', type=int, default=192,
                        help='images per batch')
    parser.add_argument('--epochs', type=int, default=50,
                        help='maximum number of epochs')
    parser.add_argument('--mean', type=int, default=183,
                        help='value to subtract from input images')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu id to run on (-1 for cpu)')
    parser.add_argument('--save', help='directory in which to save epoch snapshots.')

    parser.add_argument('train_glob', help='path to directory of images for training')
    parser.add_argument('valid_glob', help='path to directory of images for validation')

    return parser.parse_args()


args = parse_args()
permutation_indices = load_permutation_indices()

jigsaw = Jigsaw()
jigsaw.to_gpu(args.gpu)

losses = collections.deque(maxlen=20)
accs = collections.deque(maxlen=20)

optimizer = optimizers.Adam()
optimizer.setup(jigsaw)

batch_x = []
batch_y = []

for epoch in range(args.epochs):
    b = 0

    for image in glob.glob(args.train_glob):
        image = Image.open(image).convert('RGB')
        image = scale(image, min_dim=255)

        image_array = np.asarray(image)

        if len(image_array.shape) != 3:
            continue

        image_array = image_array.transpose(2,1,0) - args.mean

        crop = random_crop(image_array)

        patches = random_patches(crop)
        # patches = drop_patch(patches)   # need to change permutations for this
        patches = np.concatenate(patches).reshape(9, 3, 64, 64)

        label, permuted_patches = permute_patches(permutation_indices, patches)

        batch_y.append(label)
        batch_x.append(permuted_patches)

        if len(batch_y) == args.batch_size:
            b += 1

            batch_x = np.array(batch_x).astype(np.float32)
            batch_x = batch_x.transpose(1,0,2,3,4)  # patch dimension first
            batch_x = chainer.Variable(batch_x)
            batch_x.to_gpu(args.gpu)

            batch_y = np.array(batch_y).astype(np.int32)
            batch_y.reshape(1, -1)
            batch_y = chainer.Variable(batch_y)
            batch_y.to_gpu(args.gpu)

            loss, acc = jigsaw(batch_x, batch_y)

            optimizer.zero_grads()
            loss.backward()
            optimizer.update()

            loss.to_cpu()
            losses.append(loss.data)
            acc.to_cpu()
            accs.append(acc.data)

            if (b % 100) == 0:
                print '%d %d %.10f (%.5f) [%.5f (%.5f)]' % (
                    epoch,
                    b,
                    sum(losses) / len(losses),
                    sum(accs) / len(accs),
                    losses[-1], accs[-1],
                )

            batch_x = []
            batch_y = []

            if b >= 1000:
                break
