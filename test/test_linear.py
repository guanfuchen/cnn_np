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
from layers import linear

class TestLinear(unittest.TestCase):
    def test_forward(self):
        input_var = Variable(torch.randn(128, 20))
        linear_var = nn.Linear(20, 30)
        linear_out_var = linear_var(input_var)

        linear_var_weight_np = linear_var.weight.data.numpy()
        linear_var_bias_np = linear_var.bias.data.numpy()

        input_var_np = input_var.data.numpy()
        linear_out_var_np = linear_out_var.data.numpy()
        # print('linear_out_var_np.shape', linear_out_var_np.shape)

        linear_custom = linear.Linear(20, 30)

        np.copyto(linear_custom.weight, linear_var_weight_np)
        np.copyto(linear_custom.bias, linear_var_bias_np)

        linear_out_custom = linear_custom.forward(input_var_np)
        # print('linear_out_custom.shape', linear_out_custom.shape)


        assert abs(np.sum(linear_out_custom - linear_out_var_np)) < 0.0001

    def test_grad(self):
        input_var = Variable(torch.randn(128, 20), requires_grad=True)
        linear_var = nn.Linear(20, 30)
        linear_out_var = linear_var(input_var)

        output_var = linear_out_var.sum()
        # output_var = linear_out_var[:, 0].sum()
        output_var.backward()

        linear_var_weight_np = linear_var.weight.data.numpy()
        linear_var_bias_np = linear_var.bias.data.numpy()

        linear_var_weight_grad_np = linear_var.weight.grad.data.numpy()
        linear_var_bias_grad_np = linear_var.bias.grad.data.numpy()
        linear_var_next_eta_np = input_var.grad.data.numpy()

        input_var_np = input_var.data.numpy()
        linear_out_var_np = linear_out_var.data.numpy()
        # print('linear_out_var_np.shape', linear_out_var_np.shape)
        # print('linear_var_weight_grad_np.shape:', linear_var_weight_grad_np.shape)
        # print('linear_var_bias_grad_np.shape:', linear_var_bias_grad_np.shape)
        # print('linear_var_weight_grad_np:', linear_var_weight_grad_np)
        # print('linear_var_bias_grad_np:', linear_var_bias_grad_np)

        linear_custom = linear.Linear(20, 30)

        np.copyto(linear_custom.weight, linear_var_weight_np)
        np.copyto(linear_custom.bias, linear_var_bias_np)

        linear_out_custom = linear_custom.forward(input_var_np)
        # print('linear_out_custom.shape', linear_out_custom.shape)

        assert abs(np.sum(linear_out_custom - linear_out_var_np)) < 0.0001

        linear_eta = np.ones(linear_out_var_np.shape)
        # linear_eta = np.zeros(linear_out_var_np.shape)
        # linear_eta[:, 0] = 1

        linear_custom_next_eta = linear_custom.calc_gradient(linear_eta)
        linear_custom_weight_grad = linear_custom.weight_gradient
        linear_custom_bias_grad = linear_custom.bias_gradient

        # print('linear_custom_weight_grad.shape:', linear_custom_weight_grad.shape)
        # print('linear_custom_bias_grad.shape:', linear_custom_bias_grad.shape)
        # print('linear_custom_weight_grad:', linear_custom_weight_grad)
        # print('linear_custom_bias_grad:', linear_custom_bias_grad)

        assert abs(np.sum(linear_custom_weight_grad-linear_custom_weight_grad)) < 0.0001
        assert abs(np.sum(linear_custom_bias_grad-linear_var_bias_grad_np)) < 0.0001
        assert abs(np.sum(linear_custom_next_eta-linear_var_next_eta_np)) < 0.0001

if __name__ == '__main__':
    unittest.main()
