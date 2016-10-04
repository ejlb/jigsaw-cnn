import math
import random


def random_crop(image_arr, crop_size=225):
    m, n = image_arr.shape[1:]

    m_start = random.randint(0, m-crop_size)
    m_stop = m_start + crop_size

    n_start = random.randint(0, n-crop_size)
    n_stop = n_start + crop_size

    return image_arr[:, m_start:m_stop, n_start:n_stop]


def grid_coord_iter(grid_length, grid_square_length):
    """ Generates coordinates of grid squares.

        param: grid_length - how many squares the grid contains in one column.
        param: grid_square_length - size of each grid square
    """

    for column in range(grid_length):
        for row in range(grid_length):
            m_start = row * grid_square_length
            m_stop = (row + 1) * grid_square_length
            n_start = column * grid_square_length
            n_stop = (column + 1) * grid_square_length

            yield m_start, m_stop, n_start, n_stop


def random_patches(image_arr, patches=9, patch_size=75, crop_size=64):
    """ Split the square `image_arr` into 9 patches and then randomly crops each patch. Note
        that the following assumptions must hold:

        * `patches` is a square number
        * `image_arr` is square
        * `image_arr` dimension is `patch_size * sqrt(patches)`
    """

    random_patch_crops = []

    sq_patches = int(math.sqrt(patches))

    for m_start, m_stop, n_start, n_stop in grid_coord_iter(sq_patches, patch_size):
        patch = image_arr[:, m_start:m_stop, n_start:n_stop]
        random_patch = random_crop(patch, crop_size)
        random_patch_crops.append(random_patch)

    return random_patch_crops


def zero_patch(patches):
    """ Randomly drop / zero a patch to make reconstruction harder. To identify an
        n-permutations of we only need n-1 members """
    unlucky_patch_id = random.randint(0, len(patches))
    del patches[unlucky_patch]
    return patches
