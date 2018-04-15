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
from layers import activation

class TestReLU(unittest.TestCase):
    def test_forward(self):
        input_var = Variable(torch.randn(3, 3, 4, 4), requires_grad=True)
        relu_var = nn.ReLU()
        relu_out_var = relu_var(input_var)

        relu_out_var_np = relu_out_var.data.numpy()

        # print('relu_out_var_np.shape:', relu_out_var_np.shape)
        # print('relu_out_var_np:', relu_out_var_np)

        input_var_np = input_var.data.numpy()
        relu_custom = activation.ReLU()
        relu_out_custom = relu_custom.forward(input_var_np)
        # print('relu_out_custom.shape:', relu_out_custom.shape)
        # print('relu_out_custom:', relu_out_custom)

        assert abs(np.sum(relu_out_custom-relu_out_var_np)) < 0.0001


    def test_grad(self):
        input_var = Variable(torch.randn(3, 3, 4, 4), requires_grad=True)
        relu_var = nn.ReLU()
        relu_out_var = relu_var(input_var)

        relu_out_var_np = relu_out_var.data.numpy()

        out_var = relu_out_var.sum()
        out_var.backward()

        relu_out_var_np = relu_out_var.data.numpy()

        relu_var_next_eta = input_var.grad.data.numpy()

        # print('relu_var_next_eta:', relu_var_next_eta)

        input_var_np = input_var.data.numpy()
        # print('input_var_np:', input_var_np)
        relu_eta = np.ones(relu_out_var_np.shape)

        relu_custom = activation.ReLU()
        relu_out_custom = relu_custom.forward(input_var_np)
        # print('relu_out_custom:', relu_out_custom)

        relu_next_eta = relu_custom.calc_gradient(relu_eta)
        # print('relu_custom.indices:', relu_custom.indices)
        # print('relu_next_eta:', relu_next_eta)

        assert abs(np.sum(relu_next_eta-relu_var_next_eta)) < 0.0001


    def test_speed(self):
        pass


class TestSoftmax(unittest.TestCase):
    def test_forward(self):
        input_var = Variable(torch.randn(2, 3), requires_grad=True)
        softmax_var = nn.Softmax()
        softmax_out_var = softmax_var(input_var)

        softmax_out_var_np = softmax_out_var.data.numpy()

        # print('softmax_out_var_np.shape:', softmax_out_var_np.shape)
        # print('softmax_out_var_np:', softmax_out_var_np)

        input_var_np = input_var.data.numpy()
        softmax_custom = activation.Softmax()
        softmax_out_custom = softmax_custom.forward(input_var_np)
        # print('input_var_np:', input_var_np)
        # print('softmax_out_custom.shape:', softmax_out_custom.shape)
        # print('softmax_out_custom:', softmax_out_custom)

        assert abs(np.sum(softmax_out_custom-softmax_out_var_np)) < 0.0001


    def test_grad(self):
        input_var = Variable(torch.randn(2, 3), requires_grad=True)
        softmax_var = nn.Softmax()
        softmax_out_var = softmax_var(input_var)

        softmax_out_var_np = softmax_out_var.data.numpy()

        out_var = softmax_out_var.sum()
        # out_var = softmax_out_var[:, 0].sum()
        out_var.backward()

        softmax_var_next_eta = input_var.grad.data.numpy()

        # print('softmax_var_next_eta:', softmax_var_next_eta)

        input_var_np = input_var.data.numpy()
        # print('input_var_np:', input_var_np)
        # print('softmax_out_var_np:', softmax_out_var_np)
        softmax_eta = np.ones(softmax_out_var_np.shape)
        # softmax_eta = np.zeros(softmax_out_var_np.shape)
        # softmax_eta[:, 0] = 1

        softmax_custom = activation.Softmax()
        softmax_out_custom = softmax_custom.forward(input_var_np)
        # print('softmax_out_custom:', softmax_out_custom)

        softmax_next_eta = softmax_custom.calc_gradient(softmax_eta)
        # print('softmax_custom.indices:', softmax_custom.indices)
        # print('softmax_next_eta:', softmax_next_eta)

        assert abs(np.sum(softmax_next_eta-softmax_var_next_eta)) < 0.0001


    def test_speed(self):
        pass

class TestLogSoftmax(unittest.TestCase):
    def test_forward(self):
        input_var = Variable(torch.randn(2, 3), requires_grad=True)
        logsoftmax_var = nn.LogSoftmax()
        logsoftmax_out_var = logsoftmax_var(input_var)

        logsoftmax_out_var_np = logsoftmax_out_var.data.numpy()

        # print('logsoftmax_out_var_np.shape:', logsoftmax_out_var_np.shape)
        # print('logsoftmax_out_var_np:', logsoftmax_out_var_np)

        input_var_np = input_var.data.numpy()
        logsoftmax_custom = activation.LogSoftmax()
        logsoftmax_out_custom = logsoftmax_custom.forward(input_var_np)
        # print('input_var_np:', input_var_np)
        # print('logsoftmax_out_custom.shape:', logsoftmax_out_custom.shape)
        # print('logsoftmax_out_custom:', logsoftmax_out_custom)

        assert abs(np.sum(logsoftmax_out_custom-logsoftmax_out_var_np)) < 0.0001


    def test_grad(self):
        input_var = Variable(torch.randn(2, 3), requires_grad=True)
        logsoftmax_var = nn.LogSoftmax()
        logsoftmax_out_var = logsoftmax_var(input_var)

        logsoftmax_out_var_np = logsoftmax_out_var.data.numpy()

        out_var = logsoftmax_out_var.sum()
        # out_var = logsoftmax_out_var[:, 0].sum()
        out_var.backward()

        logsoftmax_var_next_eta = input_var.grad.data.numpy()

        # print('logsoftmax_var_next_eta:', logsoftmax_var_next_eta)

        input_var_np = input_var.data.numpy()
        # print('input_var_np:', input_var_np)
        # print('logsoftmax_out_var_np:', logsoftmax_out_var_np)
        logsoftmax_eta = np.ones(logsoftmax_out_var_np.shape)
        # logsoftmax_eta = np.zeros(logsoftmax_out_var_np.shape)
        # logsoftmax_eta[:, 0] = 1

        logsoftmax_custom = activation.LogSoftmax()
        logsoftmax_out_custom = logsoftmax_custom.forward(input_var_np)
        # print('logsoftmax_out_custom:', logsoftmax_out_custom)

        logsoftmax_next_eta = logsoftmax_custom.calc_gradient(logsoftmax_eta)
        # print('logsoftmax_custom.indices:', logsoftmax_custom.indices)
        # print('logsoftmax_next_eta:', logsoftmax_next_eta)

        assert abs(np.sum(logsoftmax_next_eta-logsoftmax_var_next_eta)) < 0.0001


    def test_speed(self):
        pass

if __name__ == '__main__':
    unittest.main()
