import glob

from PIL import Image

from chainer.dataset import DatasetMixin

import numpy as np

import patches
import permutations


class PatchDataset(DatasetMixin):
    def __init__(self,
            path_glob,
            mean,
            over_sample=1,
            drop_patch=False,
            labels=True,
        ):

        self.files = glob.glob(path_glob)
        self.n = len(self.files)
        self.mean = mean
        self.permutation_indices = permutations.load_permutation_indices()
        self.over_sample = over_sample
        self.drop_patch = drop_patch
        self.labels = labels

    def __len__(self):
        return self.n * self.over_sample

    def get_example(self, i):
        # we use each image `over_sample` times per batch with different random crops
        over_sample_index = i // self.over_sample

        image = Image.open(self.files[over_sample_index]).convert('RGB')
        image = patches.scale(image)
        image_array = np.array(image).transpose(2, 1, 0).astype(np.float32) - self.mean

        image_crop = patches.random_crop(image_array)
        image_patches = patches.random_patches(image_crop)

        if self.drop_patch:
            image_patches = patches.drop_patch(image_patches)

        image_patches = np.concatenate(image_patches).reshape(9, 3, 64, 64)

        permutated_patches, labels = permutations.permute_patches(
            self.permutation_indices,
            image_patches)

        if self.labels:
            return permutated_patches, labels
        else:
            return permutated_patches

    def get_name(self, i):
        """ file name for index `i` """
        over_sample_index = i // self.over_sample
        return self.files[over_sample_index]
