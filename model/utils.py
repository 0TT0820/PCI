# !/usr/bin/env python
# -*-coding:utf-8 -*-

import json


class TslLogEntity:

    def __init__(self):
        self.total_epoch = None
        self.epoch = None
        self.loss = None
        self.lr = None
        self.Cost = None
        self.ETA = None
        self.Criterion = {}

    def get_criterion(self):
        return self.Criterion

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_total_epoch(self, Total_epoch):
        self.total_epoch = Total_epoch

    def set_loss(self, loss):
        self.loss = loss

    def set_lr(self, lr):
        self.lr = lr

    def set_cost(self, cost):
        self.Cost = cost

    def set_eta(self, eta):
        self.ETA = eta

    def set_criterion(self, key, vallue):
        self.Criterion[key] = vallue

    def __str__(self):
        return json.dumps({
            "epoch": self.epoch,
            "total_epoch": self.total_epoch,
            "loss": self.loss,
            "lr": self.lr,
            "Cost": self.Cost,
            "ETA": self.ETA,
            "Criterion": self.Criterion
        })
