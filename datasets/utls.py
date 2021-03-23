import numpy as np
import argparse


def patch_generator(img, patch_size, stride):
    h, w, _ = img.shape
    patch_t = []
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            patch_t.append(img[i: i + patch_size, j: j + patch_size, :])
    return patch_t


def data_aug(img, mode=0):  # img: W*H
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img, axes=(0, 1))
    elif mode == 3:
        return np.flipud(np.rot90(img, axes=(0, 1)))
    elif mode == 4:
        return np.rot90(img, k=2, axes=(0, 1))
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2, axes=(0, 1)))
    elif mode == 6:
        return np.rot90(img, k=3, axes=(0, 1))
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3, axes=(0, 1)))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y1', '1', 'TRUE'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'FALSE'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
