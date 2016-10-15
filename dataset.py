import glob

from PIL import Image

import numpy as np

from chainer.dataset import DatasetMixin

import patches
import permutations


class PatchDataset(DatasetMixin):
    def __init__(self, path_glob, mean):
        self.files = glob.glob(path_glob)
        self.n = len(self.files)
        self.mean = mean
        self.permutation_indices = permutations.load_permutation_indices()

    def __len__(self):
        # TODO, make number of batches a param
        return min(self.n, 192 * 5000)

    def get_example(self, i):
        image = Image.open(self.files[i]).convert('RGB')

        image = patches.scale(image)
        image_array = np.array(image).transpose(2,1,0).astype(np.float32) - self.mean

        image_crop = patches.random_crop(image_array)

        image_patches = patches.random_patches(image_crop)
        image_patches = patches.drop_patch(image_patches)
        image_patches = np.concatenate(image_patches).reshape(9, 3, 64, 64)

        return permutations.permute_patches(self.permutation_indices, image_patches)
