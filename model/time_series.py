# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
import torch.nn as nn
from .basemodel import BaseModel


class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu",dropout=0.5, window_size=11):
        super().__init__(batch_size, device=device)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.window_size = window_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度

    def forward(self, input_seq):
        bash_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        device = f"cuda:{input_seq.get_device()}" if input_seq.get_device() >= 0 else "cpu"

        if seq_len < self.window_size:
            return torch.zeros((bash_size, 1), device=device)

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)

        input_seq = input_seq.view(bash_size, seq_len, -1)

        output, _ = self.lstm(input_seq, (h_0, c_0))
        output = output.contiguous().view(bash_size * seq_len, self.hidden_size)  #
        pred = self.linear(output)
        pred = pred.view(bash_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred

class RNN(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu", window_size=11):
        super().__init__(batch_size, device=device)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.window_size = window_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度

    def forward(self, input_seq):
        bash_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        device = f"cuda:{input_seq.get_device()}" if input_seq.get_device() >= 0 else "cpu"

        if seq_len < self.window_size:
            return torch.zeros((bash_size, 1), device=device)

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)

        input_seq = input_seq.view(bash_size, seq_len, -1)

        output, _ = self.rnn(input_seq, h_0)
        output = output.contiguous().view(bash_size * seq_len, self.hidden_size)  #
        pred = self.linear(output)
        pred = pred.view(bash_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred

class GRU(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu", window_size=11):
        super().__init__(batch_size, device=device)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.window_size = window_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(self.hidden_size, self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度
        self.activ_fun = torch.nn.Sigmoid()

    def forward(self, input_seq):
        bash_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        device = f"cuda:{input_seq.get_device()}" if input_seq.get_device() >= 0 else "cpu"

        if seq_len < self.window_size:
            return torch.zeros((bash_size, 1), device=device)

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)

        input_seq = input_seq.view(bash_size, seq_len, -1)

        output, _ = self.gru(input_seq, h_0)
        output = output.contiguous().view(bash_size * seq_len, self.hidden_size)  #
        pred = self.linear(output)
        pred = pred.view(bash_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred