#!/usr/bin/python
# -*- coding: UTF-8 -*-

import unittest
import torch
from torch import nn
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

from context import layers
from layers import conv

class TestConv2d(unittest.TestCase):
    def test_forward(self):
        input_var = Variable(torch.randn(3, 3, 5, 5))
        conv_var = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        output_var = conv_var(input_var)

        # 转换torch变量为numpy
        output_np = output_var.data.numpy()
        input_np = input_var.data.numpy()
        conv_weight_np = conv_var.weight.data.numpy()
        conv_bias_np = conv_var.bias.data.numpy()
        conv_bias_np_vector = np.reshape(conv_bias_np, (conv_bias_np.shape + (1,)))

        # print('conv_weight_np.shape:', conv_weight_np.shape)
        # print('conv_bias_np.shape:', conv_bias_np.shape)
        # print('conv_bias_np_vector.shape:', conv_bias_np_vector.shape)
        # print(output_np)

        conv_custom = conv.Conv2d(in_channels=3, out_channels=3, kernel_size=3, init_params=True)

        conv_custom_weight = conv_custom.weight
        conv_custom_bias = conv_custom.bias
        # print('conv_custom_weight.shape:', conv_custom_weight.shape)
        # print('conv_custom_bias.shape:', conv_custom_bias.shape)
        np.copyto(dst=conv_custom_weight, src=conv_weight_np)
        np.copyto(dst=conv_custom_bias, src=conv_bias_np_vector)
        assert np.array_equal(conv_custom_weight, conv_weight_np)
        assert np.array_equal(conv_custom_bias, conv_bias_np_vector)

        output_custom = conv_custom.forward(input_np)
        # print(output_custom)

        # 前向传播过程中计算结果相同
        assert abs(np.sum(output_custom-output_np)) < 0.0001


        # print(output_custom)

    def test_grad(self):
        input_var = Variable(torch.randn(3, 3, 5, 5).float(), requires_grad=True)
        conv_var = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        conv_out_var = conv_var(input_var)

        conv_out_flatten_shape = 1
        # shape[0]为batch_size
        for conv_out_var_shape in conv_out_var.data.shape[1:]:
            conv_out_flatten_shape *= conv_out_var_shape


        # output_var = conv_out_var[:, 0, 0, 0].sum()
        output_var = conv_out_var.sum()
        # print('output_var.data.shape:', output_var.data.shape)
        # print('conv_out_var.data.shape:', conv_out_var.data.shape)
        output_var.backward()
        conv_var_weight_grad = conv_var.weight.grad
        conv_var_bias_grad = conv_var.bias.grad
        conv_var_next_eta = input_var.grad

        # 转换torch变量为numpy
        input_np = input_var.data.numpy()
        conv_weight_np = conv_var.weight.data.numpy()
        conv_bias_np = conv_var.bias.data.numpy()
        conv_bias_np_vector = np.reshape(conv_bias_np, (conv_bias_np.shape + (1,)))
        conv_var_weight_grad_np = conv_var_weight_grad.data.numpy()
        conv_var_bias_grad_np = conv_var_bias_grad.data.numpy()
        conv_var_bias_grad_np_vector = np.reshape(conv_var_bias_grad_np, (conv_var_bias_grad_np.shape + (1,)))
        conv_out_var_np = conv_out_var.data.numpy()
        conv_var_next_eta_np = conv_var_next_eta.data.numpy()

        # print('conv_weight_np.shape:', conv_weight_np.shape)
        # print('conv_bias_np.shape:', conv_bias_np.shape)
        # print('conv_bias_np_vector.shape:', conv_bias_np_vector.shape)
        # print('conv_var_weight_grad_np.shape:', conv_var_weight_grad_np.shape)
        # print('conv_var_bias_grad_np.shape:', conv_var_bias_grad_np.shape)
        # print('conv_var_bias_grad_np_vector.shape:', conv_var_bias_grad_np_vector.shape)
        # print('conv_var_weight_grad_np:', conv_var_weight_grad_np)
        # print('conv_var_bias_grad_np:', conv_var_bias_grad_np)
        # print('conv_var_bias_grad_np_vector:', conv_var_bias_grad_np_vector)
        # print('conv_var_next_eta_np:', conv_var_next_eta_np)
        # print('conv_var_next_eta_np.shape:', conv_var_next_eta_np.shape)

        conv_custom = conv.Conv2d(in_channels=3, out_channels=3, kernel_size=3, init_params=True)
        conv_eta = np.ones(conv_out_var_np.shape)
        # conv_eta = np.zeros(conv_out_var_np.shape)
        # conv_eta[:, 0, 0, 0] = 1
        # print('conv_eta.shape:', conv_eta.shape)

        conv_custom_weight = conv_custom.weight
        conv_custom_bias = conv_custom.bias
        # print('conv_custom_weight.shape:', conv_custom_weight.shape)
        # print('conv_custom_bias.shape:', conv_custom_bias.shape)
        np.copyto(dst=conv_custom_weight, src=conv_weight_np)
        np.copyto(dst=conv_custom_bias, src=conv_bias_np_vector)
        assert np.array_equal(conv_custom_weight, conv_weight_np)
        assert np.array_equal(conv_custom_bias, conv_bias_np_vector)

        output_custom = conv_custom.forward(input_np)
        assert abs(np.sum(output_custom-conv_out_var_np)) < 0.0001

        # 反向传播公式
        conv_custom_next_eta = conv_custom.calc_gradient(conv_eta)
        conv_custom_weight_grad = conv_custom.weight_gradient
        conv_custom_bias_grad = conv_custom.bias_gradient
        # print('conv_next_eta:', conv_next_eta)
        # print(input_np[0, 0, 0, 0]+input_np[0, 0, 0, 1]+input_np[0, 0, 0, 2]+input_np[0, 0, 1, 0]+input_np[0, 0, 1, 1]+input_np[0, 0, 1, 2]+input_np[0, 0, 2, 0]+input_np[0, 0, 2, 1]+input_np[0, 0, 2, 2])
        # print('conv_custom_weight_grad:', conv_custom_weight_grad)
        # print('conv_custom_bias_grad:', conv_custom_bias_grad)
        assert abs(np.sum(conv_custom_weight_grad-conv_var_weight_grad_np)) < 0.0001
        assert abs(np.sum(conv_custom_bias_grad-conv_var_bias_grad_np_vector)) < 0.0001
        assert abs(np.sum(conv_custom_next_eta-conv_var_next_eta_np)) < 0.0001

    def test_speed(self):
        import time
        img = np.ones((64, 64, 3))
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, :]
        conv1 = layers.conv.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        start_time = time.time()
        conv1_forward = conv1.forward(img)
        end_time = time.time()
        print('forward time:', end_time-start_time)

        conv1_forward_real = conv1_forward.copy() + 1
        start_time = time.time()
        conv1.calc_gradient(conv1_forward_real - conv1_forward)
        conv1.backward()
        end_time = time.time()
        print('backward time:', end_time-start_time)




if __name__ == '__main__':
    unittest.main()
