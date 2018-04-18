#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

from module import Module

class RNN(Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_ih = np.zeros((self.hidden_size, self.input_size))
        self.b_ih = np.zeros(self.hidden_size)
        self.w_hh = np.zeros((self.hidden_size, self.hidden_size))
        self.b_hh = np.zeros(self.hidden_size)

        self.w_ih_gradient = np.zeros(self.w_ih.shape)
        self.b_ih_gradient = np.zeros(self.b_ih.shape)
        self.w_hh_gradient = np.zeros(self.w_hh.shape)
        self.b_hh_gradient = np.zeros(self.b_hh.shape)

    def calc_gradient(self, error):
        pass
        # for t in range(self.seq_len):
        #     for k in range(t):
        #         theta = 1
        #         for i in range(k+1, t+1):
        #             theta = np.dot(theta, np.dot(np.transpose(self.w_hh), np.diag(1-self.h_batch[i-1, 0, :])))
        #         theta *= self.input_map[k]


    def forward_rnn(self, x, h_init):
        """
        :param x: 输入seq_len*batch*input_size
        :param h_init: 隐藏batch*hidden_size
        :return: 
        """
        self.seq_len, self.batch_size, self.input_size = x.shape
        self.input_map = x
        self.h_batch = np.zeros((self.seq_len, self.batch_size, self.hidden_size))
        h_prev = h_init
        o_np = np.zeros((self.seq_len, self.batch_size, self.hidden_size))
        h_np = np.zeros((self.batch_size, self.hidden_size))
        for t in range(self.seq_len):
            x_t = x[t]
            # h_t格式batch*hidden_size
            h_t = np.tanh(np.dot(x_t, np.transpose(self.w_ih)) + np.dot(h_prev, np.transpose(self.w_hh)) + self.b_ih + self.b_hh)
            o_t = h_t
            h_prev = h_t
            o_np[t] = o_t
            h_np = h_t
            self.h_batch[t] = h_t
        return o_np, h_np


    def forward(self, x):
        pass

    def backward(self, lr=0.01):
        pass
