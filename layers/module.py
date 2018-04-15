#!/usr/bin/python
# -*- coding: UTF-8 -*-

import abc

class Module(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def calc_gradient(self, error):
        pass

    @abc.abstractmethod
    def backward(self, lr=0.01):
        pass
