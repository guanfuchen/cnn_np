#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import math

from module import Module
from utils import im2col

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, init_params=False):
        """
        :param in_channels: (int) the input channel
        :param out_channels: (int) the output channel
        :param kernel_size: (int) the kernel size
        :param stride: (int) the stirde
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_h = None
        self.input_w = None
        self.out_h = None
        self.out_w = None

        self.weight_gradient = 0
        self.bias_gradient = 0

        self.init_params = init_params

        self.weight = np.random.randn(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        self.bias = np.random.randn(self.out_channels, 1)


        # 输入图像的batch_size=N，默认为1
        self.batch_size = 1



    def forward(self, x):
        """
        :param x: (N, C_in, H_in, W_in) 通道*高度*宽度
        :return: 
        """
        self.input_map = x

        if not self.init_params:
            self.init_params = True
            weights_scale = math.sqrt(reduce(lambda x, y: x * y, self.input_map.shape) / self.out_channels)

            self.weight = np.random.standard_normal(
                size=(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)) / weights_scale
            self.bias = np.random.standard_normal(size=(self.out_channels, 1)) / weights_scale

        self.batch_size, _, self.input_h, self.input_w = x.shape

        self.out_h = (self.input_h-self.kernel_size)/self.stride + 1
        self.out_w = (self.input_w-self.kernel_size)/self.stride + 1
        # print('out_h:', self.out_h)
        # print('out_w:', self.out_w)

        # 图像转换为矩阵，N*(H*W)*(C*K*K)
        self.col_images = []

        weight_col = self.weight.reshape(self.out_channels, -1)
        # N * C_out * H_out * W_out
        conv_out = np.zeros((self.batch_size, self.out_channels, self.out_h, self.out_w))
        for batch_i in range(self.batch_size):
            # 输入的第i个图像C_in*H_in*W_in
            image_batch_i = x[batch_i, :]
            image_batch_i_col = im2col(image_batch_i, self.kernel_size, self.stride)

            self.col_images.append(image_batch_i_col)
            # print(image_batch_i_col.shape)
            # print(weight_col.shape)
            conv_out[batch_i] = np.reshape(np.dot(weight_col, np.transpose(image_batch_i_col))+self.bias, (self.out_channels, self.out_h, self.out_w))

        self.col_images = np.array(self.col_images)

        return conv_out

    # 计算梯度过程中同时将误差反向传播计算出来，根据当前误差返回上一误差
    def calc_gradient(self, error):
        self.error = error
        error_col = self.error.reshape(self.batch_size, self.out_channels, -1)
        # print('self.col_images.shape:', self.col_images.shape)
        # print('error_col.shape:', error_col.shape)
        # print('error.shape:', error.shape)

        for batch_i in range(self.batch_size):
            self.weight_gradient += np.dot(error_col[batch_i], self.col_images[batch_i]).reshape(self.weight.shape)
        # 将对应的维度相加，需要将N和最后求和
        self.bias_gradient += np.sum(error_col, axis=(0, 2)).reshape(self.bias.shape)
        # 反向传播计算上一层error

        error_pad = np.pad(self.error, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=0)
        # print('error_pad.shape:', error_pad.shape)
        # print('error:', error)
        # print('error_pad:', error_pad)

        weight_flip = self.weight[:, :, ::-1, ::-1]
        weight_flip = np.swapaxes(weight_flip, 0, 1)
        weight_flip_col = weight_flip.reshape(self.in_channels, -1)
        # print('weight_flip_col.shape:', weight_flip_col.shape)


        next_error = np.zeros((self.batch_size, self.in_channels, self.input_h, self.input_w))
        for batch_i in range(self.batch_size):
            # 输入的第i个图像C_in*H_in*W_in
            error_pad_image_batch_i = error_pad[batch_i, :]
            error_pad_image_batch_i_col = im2col(error_pad_image_batch_i, self.kernel_size, self.stride)
            # print('error_pad_image_batch_i_col.shape:', error_pad_image_batch_i_col.shape)
            next_error[batch_i] = np.reshape(np.dot(weight_flip_col, np.transpose(error_pad_image_batch_i_col)), (self.in_channels, self.input_h, self.input_w))


        # print('error_pad_image_col.shape:', error_pad_image_col.shape)
        # print('error_pad_image_col.shape:', error_pad_image_col.shape)
        # next_error = np.dot(error_pad_image_col, np.transpose(weight_flip_col)).reshape(self.batch_size, self.in_channels, self.input_h, self.input_w)
        # print('next_error.shape:', next_error.shape)
        #
        # conv_out[batch_i] = np.reshape(np.dot(weight_col, np.transpose(image_batch_i_col)) + self.bias,
        #                                (self.out_channels, self.out_h, self.out_w))

        return next_error


    def backward(self, lr=0.01):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0


if __name__ == '__main__':
    # img = misc.face()
    # plt.imshow(img)
    # plt.show()
    # channel first
    img = np.ones((64, 64, 3))
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    print(img.shape)
    conv1 = Conv2d(in_channels=3, out_channels=5, kernel_size=3)
    conv1_forward = conv1.forward(img)
    print(conv1_forward.shape)

    conv1_forward_real = conv1_forward.copy() + 1
    conv1.calc_gradient(conv1_forward_real-conv1_forward)
    # 多次梯度计算
    # conv1.calc_gradient(conv1_forward_real-conv1_forward)

    conv1.backward()

    # img_0_col = im2col(img[0, :], kernel_size=3, stride=1)
    # print(img_0_col.shape)
