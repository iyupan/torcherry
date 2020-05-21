# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020/5/21 14:16

import os

import abc

import math

import torch

from .util import ContinualTrain

from ..core import Runner


class CheckBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def check2save(self, cls:Runner, epoch: int):
        pass


class CheckBestValAcc(CheckBase):
    def __init__(self):
        self.best_val_top1 = -math.inf

    def check2save(self, cls:Runner, epoch):
        if self.best_val_top1 < cls.val_best_top_1:
            self.best_val_top1 = cls.val_best_top_1
            print("Saving Best Val Top-1 Model...")
            torch.save(cls.model.state_dict(),
                       os.path.join(cls.model_dir, 'model-nn-best-val-top-1.pt'))
            torch.save(cls.optimizer.state_dict(),
                       os.path.join(cls.model_dir, 'opt-nn-best-val-top-1.tar'))
            torch.save(cls.lr_schedule.state_dict(),
                       os.path.join(cls.model_dir, 'lr-sdl-nn-best-val-top-1.tar'))


class CheckContinueTrain(CheckBase):
    def __init__(self, continual_save_freq):
        self.continual_save_freq = continual_save_freq

    def check2save(self, cls:Runner, epoch: int):
        if epoch % self.continual_save_freq == 0:
            ContinualTrain.save(cls.model_dir, cls.log_dir, cls.model, cls.optimizer, cls.lr_schedule,
                                cls.val_best_top_1)


class CheckFrequence(CheckBase):
    def __init__(self, checkpoint_save_freq):
        self.checkpoint_save_freq = checkpoint_save_freq

    def check2save(self, cls:Runner, epoch: int):
        if epoch % self.checkpoint_save_freq == 0:
            torch.save(cls.model.state_dict(),
                       os.path.join(cls.model_dir, 'model-nn-epoch{}.pt'.format(epoch)))
            torch.save(cls.optimizer.state_dict(),
                       os.path.join(cls.model_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))
