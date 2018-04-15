#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from module import Module
from activation import LogSoftmax
from utils import get_one_hot

class CrossEntropyLoss(Module):
    def backward(self, lr=0.01):
        pass

    def calc_gradient(self, error):
        pass


    def forward(self, x):
        pass

    def forward_loss(self, input, target):
        self.input = input
        self.target = target

        self.logsoftmax_module = LogSoftmax()
        self.logsoftmax_out = self.logsoftmax_module.forward(self.input)
        self.batch_size, self.target_num = self.input.shape

        # print('target:', self.target)
        # print('batch_size:', self.batch_size)
        # print('target_num:', self.target_num)

        self.target_one_hot = get_one_hot(self.target, self.target_num)
        # print('target_one_hot:', self.target_one_hot)
        nll_log = -self.logsoftmax_out*self.target_one_hot
        # print('nll_log:', nll_log)
        return 1.0/self.batch_size * np.sum(nll_log)

    def calc_gradient_loss(self):
        error1 = -self.target_one_hot
        next_error = self.logsoftmax_module.calc_gradient(error1)
        return 1.0/self.batch_size*next_error

    def __init__(self):
        """
        交叉熵loss，loss
        """
        self.batch_size = 1
        self.target_num = 1
