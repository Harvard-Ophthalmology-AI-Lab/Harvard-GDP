import sys, os
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import random
import csv
import pickle
import statsmodels.api as sm
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

sys.path.append('.')
from utils.data_handler import *
from utils.modules import *

def find_all_files(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

def vf_to_matrix(vec, fill_in=-50):
    mat = np.empty((8,9))
    mat[:] = fill_in

    mat[0, 3:7] = vec[0:4]
    mat[1, 2:8] = vec[4:10]
    mat[2, 1:] = vec[10:18]
    mat[3, :7] = vec[18:25]
    mat[3, 8] = vec[25]
    mat[4, :7] = vec[26:33]
    mat[4, 8] = vec[33]
    mat[5, 1:] = vec[34:42]
    mat[6, 2:8] = vec[42:48]
    mat[7, 3:7] = vec[48:52]

    # mat = np.rot90(mat, k=1).copy()

    return mat

class CrossSectional_Dataset(Dataset):
    # subset: train | val | test | unmatch
    def __init__(self, data_path='./data/', subset='train', data_type='label+unlabel',
            resolution=224, need_shift=True, stretch=2.0):

        self.data_path = data_path

        self.rnflt_data_path = os.path.join(self.data_path, subset)
        self.rnflt_data = find_all_files(self.rnflt_data_path, suffix='npz')
        
        self.min_vf_val = -38.0
        self.max_vf_val = 26.0
        self.normalize_vf = 30.0

        self.dataset_len = len(self.rnflt_data)
        self.depth = 1
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch
        self.data_type = data_type

        self.unlabel_flags = np.load(os.path.join(self.data_path, 'unlabel_flags.npz'))['flags']

        if self.data_type == 'label':
            tmp_rnflt = []
            tmp_vf = []
            tmp_flags = []
            for i in range(len(self.unlabel_flags)):
                if self.unlabel_flags[i] == 0:
                    tmp_flags.append(0)
                    tmp_rnflt.append(self.rnflt_data[i])
            self.rnflt_data = tmp_rnflt
            self.unlabel_flags = tmp_flags
            self.dataset_len = len(self.rnflt_data)
        elif self.data_type == 'label+unlabel':
            pass

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        rnflt_file = os.path.join(self.rnflt_data_path, self.rnflt_data[item])
        raw_data = np.load(rnflt_file, allow_pickle=True)

        rnflt_data = raw_data['rnflt']
        rnflt_sample = np.reshape(rnflt_data, (self.size, self.size))

        rnflt_sample = np.delete(rnflt_sample, 0, 0)
        rnflt_sample = np.delete(rnflt_sample, 0, 1)
        rnflt_sample = np.repeat(rnflt_sample[:, :, np.newaxis], self.depth, axis=2)

        rnflt_sample = (np.clip(rnflt_sample, -2, 350)+2)
        rnflt_sample=rnflt_sample.astype(np.float32)
        rnflt_sample = np.transpose(rnflt_sample, [2, 0, 1])

        y = torch.tensor(raw_data['md']>=-1)
        y = y.float()
        if self.unlabel_flags is not None and self.unlabel_flags[item] == 1:
            y = torch.tensor(-100.)

        out_dict = {}

        return rnflt_sample, y, out_dict

class Longitudinal_Dataset(Dataset):
    def __init__(self, data_path='./data/', subset='train', resolution=224, outcome_type='progression_outcome_md_fast_no_p_cut',
                    unlabel_ratio=.5, data_type='label+unlabel', modality=1,
                    need_shift=True, stretch=2.0):

        self.data_path = data_path
        self.modality = modality

        self.rnflt_data_path = os.path.join(self.data_path, subset)
        self.rnflt_data = find_all_files(self.rnflt_data_path, suffix='npz')
        self.progression_type = outcome_type
        if self.progression_type == 'progression_outcome_md_fast_no_p_cut':
            self.progression_index=4
        elif self.progression_type == 'progression_outcome_td_pointwise_no_p_cut':
            self.progression_index=5

        self.min_vf_val = -38.0
        self.max_vf_val = 26.0
        self.normalize_vf = 30.0

        self.dataset_len = len(self.rnflt_data)
        self.depth = 1
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch
        self.data_type = data_type

        self.unlabel_flags = np.load(os.path.join(self.data_path, 'unlabel_flags.npz'))['flags']

        if self.data_type == 'label':
            tmp_rnflt = []
            tmp_vf = []
            tmp_flags = []
            for i in range(len(self.unlabel_flags)):
                if self.unlabel_flags[i] == 0:
                    tmp_flags.append(0)
                    tmp_rnflt.append(self.rnflt_data[i])
            self.rnflt_data = tmp_rnflt
            self.unlabel_flags = tmp_flags
            self.dataset_len = len(self.rnflt_data)
        elif self.data_type == 'label+unlabel':
            pass

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        rnflt_file = os.path.join(self.rnflt_data_path, self.rnflt_data[item])
        raw_data = np.load(rnflt_file,allow_pickle=True)

        rnflt_data = raw_data['rnflt']
        rnflt_sample = np.reshape(rnflt_data, (self.size, self.size))

        rnflt_sample = np.delete(rnflt_sample, 0, 0)
        rnflt_sample = np.delete(rnflt_sample, 0, 1)
        rnflt_sample = np.repeat(rnflt_sample[:, :, np.newaxis], self.depth, axis=2)
        
        rnflt_sample = (np.clip(rnflt_sample, -2, 350)+2)
        rnflt_sample=rnflt_sample.astype(np.float32)
        rnflt_sample = np.transpose(rnflt_sample, [2, 0, 1])

        tds_mat = None
        if self.modality == 2:
            tds_data = raw_data['tds']
            tds_data = (tds_data - self.min_vf_val) / (self.max_vf_val - self.min_vf_val) * self.stretch
            tds_mat = vf_to_matrix(tds_data, 0)
            tds_mat = np.repeat(tds_mat, 28, axis=0).repeat(25, axis=1)
            tds_mat = np.delete(tds_mat, -1, 1)
            tds_mat = tds_mat[np.newaxis, :, :].astype(np.float32)

            rnflt_sample = np.concatenate((rnflt_sample, tds_mat), axis=0)

        y = torch.tensor(float(raw_data['progression'][self.progression_index])).float()

        if self.unlabel_flags is not None and self.unlabel_flags[item] == 1:
            y = torch.tensor(-100.).float()

        out_dict = {}

        return rnflt_sample, y, out_dict


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), out_dict
