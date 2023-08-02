import numpy as np
import math
from skimage.metrics import *
from sklearn.metrics import *

import torch

def psnr(img1, img2, data_range=255.0):
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)

def ssim(img1, img2, data_range=255.0, multichannel=False):
    return structural_similarity(img1, img2, data_range=data_range, multichannel=multichannel)

# def mae(y_pred, y_gt):
#     # lower_bound = min_val/max_val
#     # upper_bound = max_val/max_val
#     # y_pred = np.clip(y_pred, lower_bound, upper_bound) * max_val
#     # y_gt = y_gt * max_val
#     mae = np.abs(y_pred - y_gt).mean()

#     return mae