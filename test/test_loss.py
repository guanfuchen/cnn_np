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
from layers import loss

class TestCrossEntropyLoss(unittest.TestCase):
    def test_forward(self):
        input_var = Variable(torch.randn(3, 5), requires_grad=True)
        target_var = Variable(torch.LongTensor(3).random_(5))
        cross_entropy_loss_var = nn.CrossEntropyLoss()
        cross_entropy_loss_out_var = cross_entropy_loss_var(input_var, target_var)

        cross_entropy_loss_out_var.backward()

        # input_var_grad_np = input_var.grad.data.numpy()
        # print('input_var_grad_np:', input_var_grad_np)

        cross_entropy_loss_out_var_np = cross_entropy_loss_out_var.data.numpy()

        # print('cross_entropy_loss_out_var_np.shape:', cross_entropy_loss_out_var_np.shape)
        # print('cross_entropy_loss_out_var_np:', cross_entropy_loss_out_var_np)

        input_var_np = input_var.data.numpy()
        target_var_np = target_var.data.numpy()
        # print('input_var_np:', input_var_np)
        # print('target_var_np:', target_var_np)

        cross_entropy_loss_custom = loss.CrossEntropyLoss()
        cross_entropy_loss_out_custom = cross_entropy_loss_custom.forward_loss(input_var_np, target_var_np)
        # print('cross_entropy_loss_out_custom.shape:', cross_entropy_loss_out_custom.shape)
        # print('cross_entropy_loss_out_custom:', cross_entropy_loss_out_custom)

        assert abs(np.sum(cross_entropy_loss_out_custom-cross_entropy_loss_out_var_np)) < 0.0001

    def test_grad(self):
        input_var = Variable(torch.randn(3, 5), requires_grad=True)
        target_var = Variable(torch.LongTensor(3).random_(5))
        cross_entropy_loss_var = nn.CrossEntropyLoss()
        cross_entropy_loss_out_var = cross_entropy_loss_var(input_var, target_var)

        cross_entropy_loss_out_var.backward()

        # input_var_grad_np = input_var.grad.data.numpy()
        # print('input_var_grad_np:', input_var_grad_np)

        cross_entropy_loss_out_var_np = cross_entropy_loss_out_var.data.numpy()
        cross_entropy_loss_grad_var_np = input_var.grad.data.numpy()

        # print('cross_entropy_loss_out_var_np.shape:', cross_entropy_loss_out_var_np.shape)
        # print('cross_entropy_loss_out_var_np:', cross_entropy_loss_out_var_np)
        # print('cross_entropy_loss_grad_var_np:', cross_entropy_loss_grad_var_np)

        input_var_np = input_var.data.numpy()
        target_var_np = target_var.data.numpy()
        # print('input_var_np:', input_var_np)
        # print('target_var_np:', target_var_np)

        cross_entropy_loss_custom = loss.CrossEntropyLoss()
        cross_entropy_loss_out_custom = cross_entropy_loss_custom.forward_loss(input_var_np, target_var_np)
        cross_entropy_loss_grad_custom = cross_entropy_loss_custom.calc_gradient_loss()
        # print('cross_entropy_loss_out_custom.shape:', cross_entropy_loss_out_custom.shape)
        # print('cross_entropy_loss_out_custom:', cross_entropy_loss_out_custom)
        # print('cross_entropy_loss_grad_custom:', cross_entropy_loss_grad_custom)

        assert abs(np.sum(cross_entropy_loss_grad_custom-cross_entropy_loss_grad_var_np)) < 0.0001



    def test_speed(self):
        pass


if __name__ == '__main__':
    unittest.main()
