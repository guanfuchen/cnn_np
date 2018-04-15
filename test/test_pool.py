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
from layers import pool

class TestMaxPool2d(unittest.TestCase):
    def test_forward(self):
        input_var = Variable(torch.randn(3, 3, 4, 4), requires_grad=True)
        maxpool_var = nn.MaxPool2d(kernel_size=2, stride=2)
        maxpool_out_var = maxpool_var(input_var)

        maxpool_out_var_np = maxpool_out_var.data.numpy()

        # print('maxpool_out_var_np:', maxpool_out_var_np)

        input_var_np = input_var.data.numpy()
        maxpool_custom = pool.MaxPool2d(kernel_size=2, stride=2)
        maxpool_out_custom = maxpool_custom.forward(input_var_np)
        # print('maxpool_out_custom:', maxpool_out_custom)

        assert abs(np.sum(maxpool_out_custom-maxpool_out_var_np)) < 0.0001


    def test_grad(self):
        input_var = Variable(torch.randn(3, 3, 4, 4), requires_grad=True)
        maxpool_var = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        maxpool_out_var, maxpool_out_indices_var = maxpool_var(input_var)

        maxpool_out_var_np = maxpool_out_var.data.numpy()

        out_var = maxpool_out_var.sum()
        out_var.backward()

        maxpool_out_var_np = maxpool_out_var.data.numpy()
        maxpool_out_indices_var_np = maxpool_out_indices_var.data.numpy()

        maxpool_var_next_eta = input_var.grad.data.numpy()

        # print('maxpool_var_next_eta:', maxpool_var_next_eta)
        # print('maxpool_out_indices_var_np:', maxpool_out_indices_var_np)
        # print('maxpool_out_indices_var_np.shape:', maxpool_out_indices_var_np.shape)

        input_var_np = input_var.data.numpy()
        # print('input_var_np:', input_var_np)
        maxpool_eta = np.ones(maxpool_out_var_np.shape)

        maxpool_custom = pool.MaxPool2d(kernel_size=2, stride=2)
        maxpool_out_custom = maxpool_custom.forward(input_var_np)
        # print('maxpool_out_custom:', maxpool_out_custom)

        maxpool_next_eta = maxpool_custom.calc_gradient(maxpool_eta)
        # print('maxpool_custom.indices:', maxpool_custom.indices)
        # print('maxpool_next_eta:', maxpool_next_eta)

        assert abs(np.sum(maxpool_next_eta-maxpool_var_next_eta)) < 0.0001


    def test_speed(self):
        pass




if __name__ == '__main__':
    unittest.main()
