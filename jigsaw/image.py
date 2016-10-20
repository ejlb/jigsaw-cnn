import random

from PIL import Image


def scale(image, min_dim=256):
    """ Aspect-ratio preserving scale such that the smallest dim is equal to `min_dim` """
    # no scaling, keep images full size
    if min_dim == -1:
        return image

    # aspect-ratio preserving scale so that the smallest dimension is `min_dim`
    width, height = image.size
    scale_dimension = width if width < height else height
    scale_ratio = float(min_dim) / scale_dimension

    if scale_ratio == 1:
        return image

    return image.resize(
        (int(width * scale_ratio), int(height * scale_ratio)),
        Image.ANTIALIAS,
    )


def random_crop(image_arr, crop_size=225):
    m, n = image_arr.shape[1:]

    m_start = random.randint(0, m-crop_size)
    m_stop = m_start + crop_size

    n_start = random.randint(0, n-crop_size)
    n_stop = n_start + crop_size

    return image_arr[:, m_start:m_stop, n_start:n_stop]
