#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from module import Module
from utils import im2col

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

        # 输入图像的batch_size=N，默认为1
        self.batch_size = 1

        self.in_channels = None
        self.out_channels = None
        self.input_h = None
        self.input_w = None
        self.out_h = None
        self.out_w = None

        self.indices = None

    def calc_gradient(self, error):
        self.error = error
        next_error = np.repeat(np.repeat(self.error , self.stride, axis=3), self.stride, axis=2) * self.indices
        return next_error

    def backward(self, lr=0.01):
        pass

    def forward(self, x):
        """
        :param x: (N, C_in, H_in, W_in) batch_size*输入通道*高度*宽度
        :return: 
        """
        self.input_map = x
        self.batch_size, self.in_channels, self.input_h, self.input_w = x.shape
        self.out_channels = self.in_channels

        self.out_h = (self.input_h-self.kernel_size)/self.stride + 1
        self.out_w = (self.input_w-self.kernel_size)/self.stride + 1

        self.indices = np.zeros(self.input_map.shape)

        pool_out = np.zeros((self.batch_size, self.out_channels, self.out_h, self.out_w))

        for batch_i in range(self.batch_size):
            # 输入的第i个图像C_in*H_in*W_in
            image_batch_i = x[batch_i, :]
            for channel_j in range(self.in_channels):
                image_batch_i_channel_j = image_batch_i[channel_j, :]
                for h_counter, h_i in enumerate(range(0, self.input_h - self.kernel_size + 1, self.stride)):
                    for w_counter, w_i in enumerate(range(0, self.input_w - self.kernel_size + 1, self.stride)):
                        image_batch_i_channel_j_patch = image_batch_i_channel_j[h_i:h_i+self.kernel_size, w_i:w_i+self.kernel_size]
                        pool_out[batch_i, channel_j, h_counter, w_counter] = np.max(image_batch_i_channel_j_patch)
                        # print('image_batch_i_channel_j_patch:', image_batch_i_channel_j_patch)
                        patch_h_max, patch_w_max = np.unravel_index(image_batch_i_channel_j_patch.argmax(), image_batch_i_channel_j_patch.shape)
                        # print('patch_h_max:', patch_h_max)
                        # print('patch_w_max:', patch_w_max)
                        self.indices[batch_i, channel_j, h_i+patch_h_max, w_i+patch_w_max] = 1

        # print('indices:', self.indices.shape)

        return pool_out
