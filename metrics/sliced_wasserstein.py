# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import scipy.ndimage

#----------------------------------------------------------------------------

def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    """
    Generate minibatch image containing minibatch image.

    Args:
        minibatch: (todo): write your description
        nhood_size: (int): write your description
        nhoods_per_image: (todo): write your description
    """
    S = minibatch.shape # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]

#----------------------------------------------------------------------------

def finalize_descriptors(desc):
    """
    Finalize the ndarray

    Args:
        desc: (array): write your description
    """
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4 # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc

#----------------------------------------------------------------------------

def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    """
    Slices the cosine mean of a.

    Args:
        A: (todo): write your description
        B: (todo): write your description
        dir_repeats: (str): write your description
        dirs_per_repeat: (str): write your description
    """
    assert A.ndim == 2 and A.shape == B.shape                           # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions
    return np.mean(results)                                             # average over repeats

#----------------------------------------------------------------------------

def downscale_minibatch(minibatch, lod):
    """
    Return the minimum scale of a minimum scale.

    Args:
        minibatch: (array): write your description
        lod: (todo): write your description
    """
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] + t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)

#----------------------------------------------------------------------------

gaussian_filter = np.float32([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]]) / 256.0

def pyr_down(minibatch): # matches cv2.pyrDown()
    """
    Pyr down an image.

    Args:
        minibatch: (float): write your description
    """
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]

def pyr_up(minibatch): # matches cv2.pyrUp()
    """
    Updates a 2dimage.

    Args:
        minibatch: (float): write your description
    """
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')

def generate_laplacian_pyramid(minibatch, num_levels):
    """
    Generate a pyramid.

    Args:
        minibatch: (float): write your description
        num_levels: (int): write your description
    """
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid

def reconstruct_laplacian_pyramid(pyramid):
    """
    Reconstruct a minimum minimum minimum minimum minimum minimum number of a minimum minimum minimum minimum minimum minimum minimum minimum minimum minimum number.

    Args:
        pyramid: (int): write your description
    """
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch

#----------------------------------------------------------------------------

class API:
    def __init__(self, num_images, image_shape, image_dtype, minibatch_size):
        """
        Initialize images.

        Args:
            self: (todo): write your description
            num_images: (int): write your description
            image_shape: (str): write your description
            image_dtype: (str): write your description
            minibatch_size: (int): write your description
        """
        self.nhood_size         = 7
        self.nhoods_per_image   = 128
        self.dir_repeats        = 4
        self.dirs_per_repeat    = 128
        self.resolutions = []
        res = image_shape[1]
        while res >= 16:
            self.resolutions.append(res)
            res //= 2

    def get_metric_names(self):
        """
        Return a list of metric names.

        Args:
            self: (todo): write your description
        """
        return ['SWDx1e3_%d' % res for res in self.resolutions] + ['SWDx1e3_avg']

    def get_metric_formatting(self):
        """
        Return the metric metric metric format.

        Args:
            self: (todo): write your description
        """
        return ['%-13.4f'] * len(self.get_metric_names())

    def begin(self, mode):
        """
        Initialize the mode.

        Args:
            self: (todo): write your description
            mode: (todo): write your description
        """
        assert mode in ['warmup', 'reals', 'fakes']
        self.descriptors = [[] for res in self.resolutions]

    def feed(self, mode, minibatch):
        """
        Generate a feed.

        Args:
            self: (todo): write your description
            mode: (todo): write your description
            minibatch: (float): write your description
        """
        for lod, level in enumerate(generate_laplacian_pyramid(minibatch, len(self.resolutions))):
            desc = get_descriptors_for_minibatch(level, self.nhood_size, self.nhoods_per_image)
            self.descriptors[lod].append(desc)

    def end(self, mode):
        """
        Calculate the mean of the descriptor.

        Args:
            self: (todo): write your description
            mode: (str): write your description
        """
        desc = [finalize_descriptors(d) for d in self.descriptors]
        del self.descriptors
        if mode in ['warmup', 'reals']:
            self.desc_real = desc
        dist = [sliced_wasserstein(dreal, dfake, self.dir_repeats, self.dirs_per_repeat) for dreal, dfake in zip(self.desc_real, desc)]
        del desc
        dist = [d * 1e3 for d in dist] # multiply by 10^3
        return dist + [np.mean(dist)]

#----------------------------------------------------------------------------
