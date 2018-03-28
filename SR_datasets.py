from __future__ import division
import torchvision
import torchvision.transforms as T
import os
import glob
import scipy.misc
import scipy.ndimage
import numpy as np
import h5py
import torch
from scipy import misc
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

class DatasetFactory(object):

    def create_dataset(self, name, root, scale=3, isTest=False):
        if name == 'VALID':
            return test_cdataset(root)
        else:
            raise Exception('Unknown dataset {}'.format(name))

class test_cdataset(Dataset):
    def __init__(self, root):
        self.root = os.path.join(root, 'DIV2K_test_LR_mild')
        self.dirlenght = 0
        for filename in os.listdir(self.root):
            if filename.endswith(".png"):
                self.dirlenght = self.dirlenght + 1
            else:
                continue

    def __len__(self):
        return self.dirlenght

    def __getitem__(self, idx):
        cnt = 0
        for filename in os.listdir(self.root):
            if filename.endswith(".png"):
                if cnt == idx:
                    low_res_img = misc.imread(os.path.join(self.root, filename))
                    break
                cnt += 1
            else:
                continue

        # h5 in numpy is (H, W, C)
        # we need to transpose to (C, H, W)
        low_res_img = np.array(low_res_img.transpose(2, 0, 1), float) / 255
        low_res_img -= 0.5

        # transform np image to torch tensor
        low_res_img = torch.Tensor(low_res_img)

        return low_res_img, filename
