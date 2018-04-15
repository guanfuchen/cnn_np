#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

from module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, init_params=False):
        """
        :param in_features: (int) 输入特征数量
        :param out_features: (int) 输出特征数量
        """
        self.in_features = in_features
        self.out_features = out_features

        self.weight_gradient = 0
        self.bias_gradient = 0

        self.init_params = init_params

        self.weight = np.random.standard_normal(size=(self.out_features, self.in_features))/100
        self.bias = np.random.standard_normal(size=self.out_features)/100

    def forward(self, x):
        """
        :param x: (N, in_features) batch_size*输入特征数量
        :return: linear_out: (N, out_features) batch_size*输出特征数量
        """
        self.input_map = x
        linear_out = np.dot(x, np.transpose(self.weight)) + self.bias
        return linear_out

    def calc_gradient(self, error):
        """
        :param error: (N, out_features) batch_size*输出特征数量
        :return: 
        """
        self.error = error
        self.weight_gradient = np.dot(np.transpose(error), self.input_map)
        self.bias_gradient = np.sum(np.transpose(self.error), axis=1)

        next_error = np.dot(error, self.weight)
        return next_error

    def backward(self, lr=0.01):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0
