# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020/5/21 00:42


import warnings

import torch.nn as nn


class CherryModule(nn.Module):
    def __init__(self):
        super(CherryModule, self).__init__()
        self.train_loader_type = None
        self.val_loader_type = None

    def tc_optimizer(self):
        warnings.warn("Optimizer in the Mould has not been defined!", UserWarning)

    def tc_train_step(self, model, data, target):
        warnings.warn("Training Step in the Mould has not been defined!", UserWarning)

    def tc_val_step(self, model, data, target):
        warnings.warn("Valing Step in the Mould has not been defined!", UserWarning)

    def tc_lr_schedule(self, optimizer):
        warnings.warn("Learning Rate Schedule in the Mould has not been defined!", UserWarning)

    def tc_train_loader(self):
        warnings.warn("TrainLoader in the Mould has not been defined!", UserWarning)

    def tc_val_loader(self):
        warnings.warn("TrainLoader in the Mould has not been defined!", UserWarning)