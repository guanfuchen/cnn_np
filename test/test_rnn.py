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
from layers import rnn

class TestRNN(unittest.TestCase):
    def test_forward(self):
        input_size = 10
        hidden_size = 20
        # 这里仅仅使用一层
        num_layers = 1
        batch_size = 3
        seq_len = 5
        # input_size，hidden_size，num_layers
        rnn_var = nn.RNN(input_size, hidden_size, num_layers)
        # 输入seq_len*batch*input_size
        input_var = Variable(torch.randn(seq_len, batch_size, input_size))
        # 隐藏num_layers * num_directions, batch, hidden_size
        h_init_var = Variable(torch.randn(num_layers, batch_size, hidden_size))
        o_n_var, h_n_var = rnn_var(input_var, h_init_var)

        o_n_var_np = o_n_var.data.numpy()
        h_n_var_np = h_n_var.data.numpy()[0] # 这里比较单层RNN
        # print('o_n_var_np:', o_n_var_np)
        # print('h_n_var_np:', h_n_var_np)
        # print('o_n_var_np.shape:', o_n_var_np.shape)
        # print('h_n_var_np.shape:', h_n_var_np.shape)

        # print(rnn_var._all_weights)
        weight_ih_l0_var = rnn_var.weight_ih_l0
        weight_hh_l0_var = rnn_var.weight_hh_l0
        bias_ih_l0_var = rnn_var.bias_ih_l0
        bias_hh_l0_var = rnn_var.bias_hh_l0

        weight_ih_l0_var_np = weight_ih_l0_var.data.numpy()
        weight_hh_l0_var_np = weight_hh_l0_var.data.numpy()
        bias_ih_l0_var_np = bias_ih_l0_var.data.numpy()
        bias_hh_l0_var_np = bias_hh_l0_var.data.numpy()

        # print('weight_ih_l0_var_np.shape:', weight_ih_l0_var_np.shape)
        # print('weight_hh_l0_var_np.shape:', weight_hh_l0_var_np.shape)
        # print('bias_ih_l0_var_np.shape:', bias_ih_l0_var_np.shape)
        # print('bias_hh_l0_var_np.shape:', bias_hh_l0_var_np.shape)


        input_var_np = input_var.data.numpy()
        h_init_var_np = h_init_var.data.numpy()[0]

        rnn_custom = rnn.RNN(input_size, hidden_size)

        np.copyto(rnn_custom.w_ih, weight_ih_l0_var_np)
        np.copyto(rnn_custom.b_ih, bias_ih_l0_var_np)
        np.copyto(rnn_custom.w_hh, weight_hh_l0_var_np)
        np.copyto(rnn_custom.b_hh, bias_hh_l0_var_np)

        o_n_custom, h_n_custom = rnn_custom.forward_rnn(input_var_np, h_init_var_np)
        # print('o_n_custom:', o_n_custom)
        # print('h_n_custom:', h_n_custom)
        # print('o_n_custom.shape:', o_n_custom.shape)
        # print('h_n_custom.shape:', h_n_custom.shape)

        assert abs(np.sum(o_n_custom-o_n_var_np)) < 0.0001
        assert abs(np.sum(h_n_custom-h_n_var_np)) < 0.0001

    def test_grad(self):
        input_size = 1
        hidden_size = 2
        # 这里仅仅使用一层
        num_layers = 1
        batch_size = 1
        seq_len = 5
        # input_size，hidden_size，num_layers
        rnn_var = nn.RNN(input_size, hidden_size, num_layers)
        # 输入seq_len*batch*input_size
        input_var = Variable(torch.randn(seq_len, batch_size, input_size))
        # 隐藏num_layers * num_directions, batch, hidden_size
        h_init_var = Variable(torch.randn(num_layers, batch_size, hidden_size))
        o_n_var, h_n_var = rnn_var(input_var, h_init_var)



        o_n_var_np = o_n_var.data.numpy()
        h_n_var_np = h_n_var.data.numpy()[0]  # 这里比较单层RNN
        # print('o_n_var_np:', o_n_var_np)
        # print('h_n_var_np:', h_n_var_np)
        # print('o_n_var_np.shape:', o_n_var_np.shape)
        # print('h_n_var_np.shape:', h_n_var_np.shape)

        # print(rnn_var._all_weights)
        weight_ih_l0_var = rnn_var.weight_ih_l0
        weight_hh_l0_var = rnn_var.weight_hh_l0
        bias_ih_l0_var = rnn_var.bias_ih_l0
        bias_hh_l0_var = rnn_var.bias_hh_l0

        weight_ih_l0_var_np = weight_ih_l0_var.data.numpy()
        weight_hh_l0_var_np = weight_hh_l0_var.data.numpy()
        bias_ih_l0_var_np = bias_ih_l0_var.data.numpy()
        bias_hh_l0_var_np = bias_hh_l0_var.data.numpy()

        loss = o_n_var.sum()
        loss.backward()

        weight_ih_l0_var_grad_np = weight_ih_l0_var.grad.data.numpy()
        weight_hh_l0_var_grad_np = weight_hh_l0_var.grad.data.numpy()
        bias_ih_l0_var_grad_np = bias_ih_l0_var.grad.data.numpy()
        bias_hh_l0_var_grad_np = bias_hh_l0_var.grad.data.numpy()

        # print('weight_ih_l0_var_grad_np:', weight_ih_l0_var_grad_np)

        # print('weight_ih_l0_var_np.shape:', weight_ih_l0_var_np.shape)
        # print('weight_hh_l0_var_np.shape:', weight_hh_l0_var_np.shape)
        # print('bias_ih_l0_var_np.shape:', bias_ih_l0_var_np.shape)
        # print('bias_hh_l0_var_np.shape:', bias_hh_l0_var_np.shape)


        input_var_np = input_var.data.numpy()
        h_init_var_np = h_init_var.data.numpy()[0]

        rnn_custom = rnn.RNN(input_size, hidden_size)

        np.copyto(rnn_custom.w_ih, weight_ih_l0_var_np)
        np.copyto(rnn_custom.b_ih, bias_ih_l0_var_np)
        np.copyto(rnn_custom.w_hh, weight_hh_l0_var_np)
        np.copyto(rnn_custom.b_hh, bias_hh_l0_var_np)

        o_n_custom, h_n_custom = rnn_custom.forward_rnn(input_var_np, h_init_var_np)
        # print('o_n_custom:', o_n_custom)
        # print('h_n_custom:', h_n_custom)
        # print('o_n_custom.shape:', o_n_custom.shape)
        # print('h_n_custom.shape:', h_n_custom.shape)


    def test_speed(self):
        pass


if __name__ == '__main__':
    unittest.main()
