import numpy as np


def mask_along_dim(y, dim, size=0.2):
    mask = np.ones(y.shape, dtype=np.float32)
    size = np.random.randint(low=0, high=int(y.shape[dim] * size))
    start_pos = np.random.randint(low=0, high=y.shape[dim] - size)
    if dim == 0:
        mask[start_pos:start_pos + size, :] = 0
    else:
        mask[:, start_pos:start_pos + size] = 0
    return y * mask


def spec_augment(y, num=3):
    num = np.random.randint(0, 1 + num)
    for i in range(num):
        dim = np.random.randint(0, 2)
        y = mask_along_dim(y, dim)
    return y
