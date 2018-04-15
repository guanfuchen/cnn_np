#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


from layers.module import Module
from layers.utils import im2col
from layers import conv, pool, activation, linear, loss

if __name__ == '__main__':
    batch_size = 100
    input_np = np.random.rand(batch_size, 1, 28, 28)
    target_np = np.random.randint(10, size=batch_size)
    print('input_np.shape:', input_np.shape)
    print('target_np.shape:', target_np.shape)
    conv1 = conv.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
    maxpool1 = pool.MaxPool2d(kernel_size=2, stride=2)
    relu1 = activation.ReLU()
    conv2 = conv.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
    maxpool2 = pool.MaxPool2d(kernel_size=2, stride=2)
    relu2 = activation.ReLU()

    # fc1 = linear.Linear(out_flatten.shape[1], 500)
    fc1 = linear.Linear(800, 500)
    fc2 = linear.Linear(500, 10)
    loss_layer = loss.CrossEntropyLoss()


    for epoch in range(1000):
        out = conv1.forward(input_np)
        out = maxpool1.forward(out)
        out = relu1.forward(out)
        out = conv2.forward(out)
        out = maxpool2.forward(out)
        out = relu2.forward(out)
        # print('out.shape:', out.shape)
        reshape_to_flatten = out.shape
        out_flatten = out.reshape(batch_size, -1)
        # print('out_flatten.shape:', out_flatten.shape)

        out = fc1.forward(out_flatten)
        out = fc2.forward(out)

        # print('out:', out)
        # print('out.shape:', out.shape)
        loss_val = loss_layer.forward_loss(out, target_np)
        print('loss_val:', loss_val)

        theta = loss_layer.calc_gradient_loss()
        theta = fc2.calc_gradient(theta)
        theta = fc1.calc_gradient(theta)
        theta = theta.reshape(reshape_to_flatten)
        # print('theta:', theta)
        theta = maxpool2.calc_gradient(theta)
        theta = conv2.calc_gradient(theta)
        theta = relu1.calc_gradient(theta)
        theta = maxpool1.calc_gradient(theta)
        theta = conv1.calc_gradient(theta)

        fc2.backward()
        fc1.backward()
        relu2.backward()
        maxpool2.backward()
        conv2.backward()
        relu1.backward()
        maxpool1.backward()
        conv1.backward()
