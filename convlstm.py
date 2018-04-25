#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
from torch import nn
from torch.autograd import Variable

# 代码来源[convlstm.py](https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py)
# [Convolution_LSTM_PyTorch](https://github.com/automan000/Convolution_LSTM_PyTorch)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        """
        input_size: （height, width）输入2D高度和宽度
        input_dim: 输入维度
        hidden_dim: hidden_dim隐藏层维度
        kernel_size：卷积核大小
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        h_init = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        c_init = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        return h_init, c_init


class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        """
        input_size: （height, width）输入2D高度和宽度
        input_dim: 输入维度
        hidden_dim: [dim1, dim2, dim3]隐藏层维度
        kernel_size：卷积核大小
        """
        super(ConvLSTM, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(self.hidden_dim)
        self.kernel_size = [kernel_size] * self.num_layers

        cell_list = []
        for i in range(0, self.num_layers):
            # 第一层是input_dim
            if i == 0:
                cur_input_dim = self.input_dim
            else:
                cur_input_dim = self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width), input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i], kernel_size=self.kernel_size[i]))

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, x):
        """
        :param x: (t, b, c, h, w)
        :return:
        """
        x.permute(1, 0, 2, 3, 4) # (t, b, c, h, w) to (b, t, c, h, w)
        batch_size, seq_len,  _, _, _ = x.size()
        hidden_state = self.init_hidden(batch_size=batch_size)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list[-1], last_state_list[-1]

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


if __name__ == '__main__':
    # convlstm = ConvLSTM(input_size=(32, 32), input_dim=3, hidden_dim=[64, 64, 128], kernel_size=(3, 3), num_layers=3, batch_first=True, bias=True, return_all_layers=False)
    convlstm = ConvLSTM(input_size=(64, 64), input_dim=3, hidden_dim=[64, 64, 128], kernel_size=(3, 3))
    input_x = Variable(torch.randn(5, 2, 3, 64, 64)) # (t, b, c, h, w)
    layer_output_list, last_state_list = convlstm(input_x)
    # print('layer_output_list:', layer_output_list)
    # print('last_state_list:', last_state_list)
