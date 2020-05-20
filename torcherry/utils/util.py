# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2019-09-05 19:39


import os
import time

import copy
import pickle
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm


class ContinualTrain(object):
    save_info_name = "keep_training_info.pkl"
    save_model_name = "model-nn-keep.pt"
    save_optimizer_name = "opt-nn-checkpoint-keep.tar"
    save_lr_schedule_name = "lr-sdl-nn-checkpoint-keep.tar"

    @classmethod
    def save(cls, model_dir, log_dir, model, optimizer, lr_schedule, best_top_1):
        end_time = time.time()
        save_model_path = os.path.join(model_dir, cls.save_model_name)
        save_optimizer_path = os.path.join(model_dir, cls.save_optimizer_name)
        save_lr_schedule_path = os.path.join(model_dir, cls.save_lr_schedule_name)
        save_info = dict(
            model_dir=model_dir,
            log_dir=log_dir,
            save_model_path=save_model_path,
            save_optimizer_path=save_optimizer_path,
            save_lr_schedule_path=save_lr_schedule_path,
            best_top_1 = best_top_1,
        )

        with open(os.path.join(model_dir, cls.save_info_name), "wb") as f:
            pickle.dump(save_info, f)

        torch.save(model.state_dict(), save_model_path)
        torch.save(optimizer.state_dict(), save_optimizer_path)
        torch.save(lr_schedule.state_dict(), save_lr_schedule_path)

    @classmethod
    def read(cls, model_dir):
        with open(os.path.join(model_dir, cls.save_info_name), "rb") as f:
            save_info = pickle.load(f)

        model_dir = save_info["model_dir"]
        log_dir = save_info["log_dir"]
        save_model_path = save_info["save_model_path"]
        save_optimizer_path = save_info["save_optimizer_path"]
        save_lr_schedule_path = save_info["save_lr_schedule_path"]
        best_top_1 = save_info["best_top_1"]

        return model_dir, log_dir, save_model_path, save_optimizer_path, save_lr_schedule_path, best_top_1


# Solve the problem that model with single gpu load checkpoints of parallel models
def load_model(is_multi_gpus: bool, model: nn.Module, model_path: str):
    pretrained_dict = torch.load(model_path)
    is_checkpoint_parallel = list(pretrained_dict.keys())[0].startswith('module.')

    if is_multi_gpus:
        print("DataParallel...")
        if is_checkpoint_parallel:
            model = nn.DataParallel(model)
            model.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(pretrained_dict)
            model = nn.DataParallel(model)
    else:
        if is_checkpoint_parallel:
            new_key = dict()
            for k, v in pretrained_dict.items():
                new_key[k[7:]] = v
            model.load_state_dict(new_key)
        else:
            model.load_state_dict(pretrained_dict)

    return model


def num_para_calcular(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
    #     print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
    #     print("该层参数和：" + str(l))
        k = k + l
    print(r"Total Params：" + str(k))


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


def create_nonexistent_folder(abs_path):
    if os.path.exists(abs_path):
        raise FileExistsError("Folder exists.")
    else:
        os.makedirs(abs_path)


def eval_model_by_dali(model, device, loader, metrics, gpu_num=1):

    metrics_ = copy.deepcopy(metrics)

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for data_pairs in loader:
                for data_pair in data_pairs:
                    data = data_pair["data"].to(device, non_blocking=True)
                    target = data_pair["label"].squeeze().long().to(device, non_blocking=True).view(-1)
                    output_logits = model(data)

                    for metric in metrics_:
                        metric.add_metric_record(output_logits, target)

                    pbar.update(1)

    return metrics_


def eval_model_by_torch(model, device, loader, metrics):

    metrics_ = copy.deepcopy(metrics)

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for data_pair in loader:
                data = data_pair[0].to(device, non_blocking=True)
                target = data_pair[1].squeeze().long().to(device, non_blocking=True).view(-1)
                output_logits = model(data)

                for metric in metrics_:
                    metric.add_metric_record(output_logits, target)

                pbar.update(1)

    return metrics_


