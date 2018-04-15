#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from module import Module

class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x):
        self.input_map = x
        relu_out = np.maximum(x, 0)
        return relu_out

    def calc_gradient(self, error):
        self.error = error
        next_error = np.copy(error)
        next_error[self.input_map<0]=0
        return next_error

    def backward(self, lr=0.01):
        pass

class Softmax(Module):
    def __init__(self):
        self.batch_size = 1
        self.in_features = 1
        self.out_features = 1
        self.softmax_out = None
        pass

    def forward(self, x):
        self.input_map = x
        self.batch_size, self.in_features = x.shape
        self.out_features = self.in_features
        self.softmax_out = np.zeros((self.batch_size, self.out_features))
        for batch_i in range(self.batch_size):
            x_batch_i = x[batch_i, :]
            x_batch_i_exp = np.exp(x_batch_i)
            x_batch_i_exp_sum = sum(x_batch_i_exp)
            self.softmax_out[batch_i, :] = x_batch_i_exp / x_batch_i_exp_sum
        return self.softmax_out


    def calc_gradient(self, error):
        self.error = error
        next_error = np.zeros(self.input_map.shape)
        gradient_ij = np.zeros((self.in_features, self.in_features))
        for batch_i in range(self.batch_size):
            error_batch_i = error[batch_i, :]
            softmax_out_batch_i = self.softmax_out[batch_i, :]
            for i in range(self.in_features):
                for j in range(self.in_features):
                    if i==j:
                        gradient_ij[i, j] = softmax_out_batch_i[i] * (1-softmax_out_batch_i[j])
                    else:
                        gradient_ij[i, j] = - softmax_out_batch_i[i] * softmax_out_batch_i[j]
            next_error[batch_i] = np.dot(gradient_ij, error_batch_i)
            # print('gradient_ij:', gradient_ij)
            # print('error_batch_i:', error_batch_i)
        return next_error

    def backward(self, lr=0.01):
        pass

class LogSoftmax(Module):
    def __init__(self):
        self.batch_size = 1
        self.in_features = 1
        self.out_features = 1
        self.softmax_out = None
        self.logsoftmax_out = None
        pass

    def forward(self, x):
        self.input_map = x
        self.softmax_module = Softmax()
        self.softmax_out = self.softmax_module.forward(x)
        self.logsoftmax_out = np.log(self.softmax_out)
        return self.logsoftmax_out


    def calc_gradient(self, error):
        self.error = error
        error_1 = 1.0/self.softmax_out * error
        next_error = self.softmax_module.calc_gradient(error_1)
        return next_error

    def backward(self, lr=0.01):
        pass
