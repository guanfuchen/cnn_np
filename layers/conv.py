#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

class Conv2d(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
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

        self.weight = np.random.standard_normal(size=(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = np.random.standard_normal(size=(out_channels, 1))



        # 输入图像的batch_size=N，默认为1
        self.batch_size = 1



    def forward(self, x):
        """
        :param x: (N, C_in, H_in, W_in) 通道*高度*宽度
        :return: 
        """
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
        error_col = error.reshape(self.batch_size, self.out_channels, -1)
        print(self.col_images.shape)
        print(error_col.shape)

        for batch_i in range(self.batch_size):
            self.weight_gradient += np.dot(error_col[batch_i], self.col_images[batch_i]).reshape(self.weight.shape)
        # 将对应的维度相加
        self.bias_gradient += np.sum(error_col, axis=(0, 2)).reshape(self.bias.shape)
        # TODO
        # 反向传播计算上一层error

    def backward(self, lr=0.01):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0


def im2col(img, kernel_size, stride=1):
    """
    :param img: 输入的图像 C_in H_in W_in
    :param kernel_size: 卷积核大小
    :param stride: 卷积核间距
    :return: img_cols (H*W) * (C*K*K)
    """
    img_channel, img_h, img_w = img.shape
    # img_cols_lists = []
    img_cols = None
    # print('img_h', img_h)
    # print('img_w', img_w)
    for channel_i in range(img_channel):
        # 通道i的图像是 H W
        img_channel_i = img[channel_i, :]
        img_channel_i_cols = []
        for h_i in range(0, img_w-kernel_size+1, stride):
            for w_i in range(0, img_h-kernel_size+1, stride):
                img_channel_i_patch = img_channel_i[h_i:h_i+kernel_size, w_i:w_i+kernel_size]
                # print(img_channel_i_patch.shape)
                # 小的patch K*K reshape为行向量
                img_channel_i_patch_row = img_channel_i_patch.reshape([-1])
                img_channel_i_cols.append(img_channel_i_patch_row)
                # print(img_channel_i_patch_row.shape)
                assert img_channel_i_patch_row.shape ==  (kernel_size*kernel_size, )
        # print('len(img_channel_i_cols):', len(img_channel_i_cols))
        img_channel_i_cols = np.array(img_channel_i_cols)
        # print('img_channel_i_cols.shape:', img_channel_i_cols.shape)
        # if not img_cols_lists:
        #     img_cols = img_channel_i_cols
        # else:
        #     img_cols = np.hstack(img_cols, img_channel_i_cols)
        # img_cols_lists.append(img_channel_i_cols)
        if img_cols is None:
            img_cols = img_channel_i_cols
        else:
            img_cols = np.hstack((img_cols, img_channel_i_cols))

    return img_cols


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
