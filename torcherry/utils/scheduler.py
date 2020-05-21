# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020/5/21 15:40

from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler


class WarmMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warm_init=0.0001, warm_epoch=10, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.warm_init = warm_init
        self.warm_epoch = warm_epoch

        super(WarmMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_epoch:
            return  [self.warm_init + self.last_epoch*(base_lr / self.warm_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                    for base_lr in self.base_lrs]


class SteadyWarmMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warm_init=0.0001, steady_epoch=None, warm_epoch=10, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.warm_init = warm_init

        self.warm_epoch = warm_epoch
        if steady_epoch is None:
            self.steady_epoch = int(self.warm_epoch / 2)
        elif steady_epoch > self.warm_epoch:
            raise ValueError("Steady_Epoch should not larger than Warm_Epoch!")
        else:
            self.steady_epoch = steady_epoch

        super(SteadyWarmMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.steady_epoch:
            return [self.warm_init for _ in self.base_lrs]
        if self.last_epoch < self.warm_epoch:
            return  [self.warm_init + (self.last_epoch - self.steady_epoch)*(base_lr / (self.warm_epoch - self.steady_epoch)) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                    for base_lr in self.base_lrs]