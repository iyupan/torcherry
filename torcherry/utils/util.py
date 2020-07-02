# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2019-09-05 19:39


import os
import time
import warnings

import copy
import random
import pickle

import numpy as np


import torch
import torch.nn as nn

from ..module import CherryModule


def set_env_seed(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

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
def load_model(is_multi_gpus: bool, use_gpu: bool, model: CherryModule, model_path: str):
    pretrained_dict = torch.load(model_path)
    is_checkpoint_parallel = list(pretrained_dict.keys())[0].startswith('module.')

    if is_multi_gpus and use_gpu:
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


def create_nonexistent_folder(abs_path):
    if os.path.exists(abs_path):
        warnings.warn("Folder %s exists." % abs_path)
    else:
        os.makedirs(abs_path)

