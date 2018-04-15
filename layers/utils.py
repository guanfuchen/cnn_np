#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

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
        for h_i in range(0, img_h-kernel_size+1, stride):
            for w_i in range(0, img_w-kernel_size+1, stride):
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

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]
