"""
@Author: Yiting CHEN
@Time: 2022/11/9 上午12:17
@Email: chenyiting@whu.edu.cn
version: python 3.9
Created by PyCharm
"""

import os
from abc import ABC
import torch
import torch.utils.data
import numpy as np
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import glob
import random
from trainOption import TrainOptions


class DatasetBase(torch.utils.data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.rgb_train_files = glob.glob(os.path.join(self.opt.dataroot, self.opt.train, "*", "*.jpg"))
        self.depth_train_files = [f.replace('.jpg', '.png') for f in self.rgb_train_files]
        self.length = len(self.rgb_train_files)
        self.rand_zoom = self.opt.rand_zoom
        self.rand_flip = self.opt.if_flip
        # self.rand_rotate = self.opt.rand_rotate
        self.load_size = self.opt.load_size
        self.load_scale = min(self.opt.load_size[0]/480, self.opt.load_size[1]/640)


    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __len__(self):
        return self.length

    def get_rgb_image(self, idx):
        raise NotImplementedError()

    def get_depth_image(self, idx):
        raise NotImplementedError()

    def random_flip(self, rgb, depth):
        if random.random() > 0.5:
            rgb = tf.hflip(rgb)
            depth = tf.hflip(depth)
        if random.random() > 0.5:
            rgb = tf.vflip(rgb)
            depth = tf.vflip(depth)
        return rgb, depth

    def random_rotate(self, rgb, depth):
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            rgb = tf.rotate(rgb, angle)
            depth = tf.rotate(depth, angle)
        return rgb, depth

    def random_zoom(self, rgb, depth):
        zoom_level = np.random.uniform(0.6, 1.0, size=[2])
        iw, ih = rgb.shape[1], rgb.shape[2]

        rgb = tf.center_crop(rgb, [int(round(iw * zoom_level[0])), int(round(ih * zoom_level[1]))])
        depth = tf.center_crop(depth, [int(round(iw * zoom_level[0])), int(round(ih * zoom_level[1]))])
        return rgb, depth

    def set_test_mode(self):
        """ no data augmentation """
        self.rand_zoom = 0
        self.rand_flip = 0

    def __getitem__(self, idx):
        rgb_img = self.get_rgb_image(idx)
        depth_img = self.get_depth_image(idx)
        rgb_file_path = self.rgb_train_files[idx]
        depth_file_path = self.depth_train_files[idx]

        return rgb_img, depth_img, rgb_file_path, depth_file_path


import imageio.v2 as imageio
from PIL import Image


class nyu_dataset(DatasetBase):
    def __init__(self, opt):
        super(nyu_dataset, self).__init__(opt)

    def get_depth_path(self, idx):
        return self.depth_train_files[idx]

    def get_rgb_path(self, idx):
        return self.rgb_train_files[idx]

    def get_depth_image(self, idx):
        data_path = self.get_depth_path(idx)
        depth_img = imageio.imread(data_path)
        depth_img = np.asarray(depth_img)

        return self.numpy_to_torch(depth_img)

    def get_rgb_image(self, idx):
        data_path = self.get_rgb_path(idx)
        rgb_img = imageio.imread(data_path)
        rgb_img = np.asarray(rgb_img) / 255.
        return self.numpy_to_torch(rgb_img.transpose((2, 0, 1)))

    def __getitem__(self, idx):
        depth_img = self.get_depth_image(idx)
        depth_img_path = self.get_depth_path(idx)

        rgb_img = self.get_rgb_image(idx)
        rgb_img_path = self.get_rgb_path(idx)

        if self.rand_zoom:
            rgb_img, depth_img = self.random_zoom(rgb_img, depth_img)
        if self.rand_flip:
            rgb_img, depth_img = self.random_flip(rgb_img, depth_img)

        aug_rgb_img = tf.resize(rgb_img, self.load_size)
        aug_depth_img = tf.resize(depth_img, self.load_size)

        return {'rgb': aug_rgb_img, 'depth': aug_depth_img, 'rgb_path': rgb_img_path, 'depth_path': depth_img_path}



if __name__ == "__main__":
    from torchvision.transforms import ToPILImage
    import matplotlib.pyplot as plt
    opt = TrainOptions().parse()  # get training options
    data_loader = nyu_dataset(opt)
    print(len(data_loader))
    data = data_loader[0]
    print(data['rgb'].shape)
    plt.imshow(data['rgb'].permute(1, 2, 0))
    plt.show()
    plt.imshow(data['depth'].permute(1, 2, 0))
    plt.show()
