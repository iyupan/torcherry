# -*- coding: UTF-8 -*-

# Author: Perry
# @Time: 2019/10/24 17:51


import abc

import torch


class MetricBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_metric_record(self, logits, target, loss):
        pass

    @abc.abstractmethod
    def get_metric(self):
        pass


class MetricAccuracy(MetricBase):
    def __init__(self, top_k):
        self.top_k = top_k
        self.correct_num = 0
        self.data_num = 0

    def add_metric_record(self, logits, target, loss):
        self.data_num += logits.size(0)

        target_resize = target.view(-1, 1)
        _, pred = logits.topk(self.top_k, 1, True, True)
        self.correct_num += torch.eq(pred, target_resize).sum().float().item()

    def get_metric(self):
        if self.data_num:
            return dict(metric_name="top_%d" % self.top_k, metric_value=self.correct_num / self.data_num)
        else:
            raise ValueError("The parameter data_num should not be None.")


class MetricLoss(MetricBase):
    def __init__(self):
        self.loss = 0
        self.data_num = 0

    def add_metric_record(self, logits, target, loss):
        self.data_num += logits.size(0)

        self.loss += loss.item()

    def get_metric(self):
        if self.data_num:
            return dict(metric_name="loss", metric_value=self.loss / self.data_num)
        else:
            raise ValueError("The parameter data_num should not be None.")
