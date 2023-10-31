# !/usr/bin/env python
# -*-coding:utf-8 -*-

import json

import sklearn.metrics

# try:
from tensorflow.python.keras.callbacks import CallbackList
# except ImportError:
#     from tensorflow.python.keras._impl.keras.callbacks import CallbackList
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import *
import time
from .callbacks import History
from model.utils import TslLogEntity

class BaseModel(nn.Module):
    def __init__(self, bashsize, device='cpu'):

        super(BaseModel, self).__init__()
        self.lr = None
        self.metrics_names = None
        self.optim = None
        self.loss_func = None
        self.metrics = None
        self.bashsize = bashsize
        self.device = device
        self.history = History()

    def fit(self, train, batch_size=None, epochs=1, verbose=1, initial_epoch=0,
            validation_data=None, callbacks=None):

        optimizer = self.optim
        loss_func = self.loss_func

        steps_per_epoch = len(list(train))
        sample_num = steps_per_epoch * batch_size  # 这里还需要改

        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False
        train_start_time = time.time()
        train_loss_list = []
        test_loss_list = []
        for epoch in range(epochs):
            model = self.train()
            callbacks.on_epoch_begin(epoch)
            log_etity = TslLogEntity()
            total_loss_epoch = 0
            test_loss_epoch = 0
            for (seq, label) in train:
                seq = seq.to(self.device)
                label = label.to(self.device)
                y_pred = model(seq)
                # weight = torch.tensor([5 if x.cpu() > 0 else 1 for x in label]).to(self.device)
                loss = loss_func(y_pred, label)

                total_loss_epoch += loss.item()

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                test_sample_num = len(list(validation_data)) * batch_size  # 这里还需要改
                for (seq, label) in validation_data:
                    seq = seq.to(self.device)
                    label = label.to(self.device)
                    y_pred = model(seq)
                    loss = loss_func(y_pred, label)
                    test_loss_epoch += loss.item()

            log_etity.set_epoch(epoch)
            log_etity.set_total_epoch(epochs)
            log_etity.set_cost(int(time.time() - train_start_time))
            log_etity.set_eta(int((epochs - epoch) * (time.time() - train_start_time) / (epoch + 1)))
            log_etity.set_loss(total_loss_epoch / sample_num)
            log_etity.set_lr(self.lr)
            train_loss_list.append(total_loss_epoch / sample_num)
            test_loss_list.append(test_loss_epoch/ test_sample_num)
            train_result = self.evaluate(validation_data=train)
            for name, result in train_result.items():
                log_etity.set_criterion("train_" + name, result)

            if validation_data is not None:
                eval_result = self.evaluate(validation_data=validation_data)
                for name, result in eval_result.items():
                    log_etity.set_criterion("val_" + name, result)
                print(log_etity)

            callbacks.on_epoch_end(epoch, log_etity.Criterion)
        callbacks.on_train_end()
        return train_loss_list, test_loss_list

    def evaluate(self, validation_data):
        """

        :param validation_data: dataloader date.
        :return: Dict contains metric names and metric values.
        """
        # x = list(validation_data)[0]
        #
        # pred_ans = self.predict(x, batch_size)
        # eval_result = {}
        # for name, metric_fun in self.metrics.items():
        #     eval_result[name] = metric_fun(y, pred_ans)
        # return eval_result

        model = self.eval()
        pred_ans = []
        y = []
        for x_, y_ in validation_data:
            x = x_.to(self.device).float()
            y_pred = model(x).cpu().data.numpy()  # .squeeze()
            pred_ans.append(y_pred)
            y.append(y_)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(np.concatenate(y).astype("float64"),
                                           np.concatenate(pred_ans).astype("float64"))

        return eval_result

    def predict(self, x):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()

        test_loader = x

        pred_ans = []
        true_label = []
        with torch.no_grad():
            for (seq, label) in test_loader:
                if bash_size == 0:
                    bash_size = len(seq)
                x = seq.to(self.device).float()
                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)
                true_label.append(label.data.numpy())
        return np.concatenate(pred_ans).astype("float64"), np.concatenate(true_label)

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                lr=0.001
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        :param lr: learning rate
        """
        self.metrics_names = ["loss"]
        self.lr = lr
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)


    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=self.lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=self.lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=self.lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=self.lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def get_accuracy_score(self, y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def get_recall_score(self, y_true, y_pred):
        return recall_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def get_rmse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "rmse":
                    metrics_[metric] = self.get_rmse
                if metric == "mae":
                    metrics_[metric] = mean_absolute_error
                if metric == "p_mse":
                    metrics_[metric] = mean_absolute_percentage_error
                if metric == 'r2_score':
                    metrics_[metric] = r2_score
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self.get_accuracy_score
                if metric == "recall_score":
                    metrics_[metric] = self.get_recall_score
                self.metrics_names.append(metric)
        return metrics_
