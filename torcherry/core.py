# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020/5/21 01:05

import os
import time
import math

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from tqdm import tqdm

from . import CherryModule

from .utils.util import load_model, eval_model_by_dali, ContinualTrain, create_nonexistent_folder


class Runner(object):
    def __init__(self):
        pass

    def fit(self, model: CherryModule, device, train_loader=None, optimizer=None, lr_schedule=None, step_func=None,
            save_path="./save", tensorboard=True, train_callbacks=None,
            continual_train_model_dir=None, record_setting: str = None, pre_train_model_path=None, train_epochs=0):
        gpu_num = torch.cuda.device_count()
        multi_gpus = gpu_num > 1

        self.model = model
        self.device = device

        self.model.to(device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = model.tc_optimizer()

        if lr_schedule:
            self.lr_schedule = lr_schedule(self.optimizer)
        else:
            self.lr_schedule = model.tc_lr_schedule(self.optimizer)

        if step_func:
            self.step_func = step_func
        else:
            self.step_func = model.tc_train_step

        if train_loader:
            self.train_loader = train_loader()
        else:
            self.train_loader = model.tc_train_loader()

        self.save_path = save_path

        # ----------------- Continual Training -------------------------
        # Load Model, Optimizer
        # Set model_dir, log_dir, and start_epoch

        self.best_top_1 = - math.inf
        self.start_epoch = 0

        if continual_train_model_dir:
            continual_info = ContinualTrain.read(continual_train_model_dir)
            model_dir, log_dir, model_path, optimizer_path, lr_schedule_path, best_top_1 = continual_info

            self.optimizer.load_state_dict(torch.load(optimizer_path))
            self.lr_schedule.load_state_dict(torch.load(lr_schedule_path))

            print("Loading the Model of Epoch %d..." % self.lr_schedule.last_epoch)
            self.model = load_model(multi_gpus, self.model, model_path)

            self.start_epoch = self.lr_schedule.last_epoch

        else:

            # ------------------- Set Model-Save Path ----------------------

            model_root = self.save_path
            if not os.path.exists(model_root):
                os.makedirs(model_root)

            # ------------------- Save Running Setting ----------------------
            if record_setting:
                with open(os.path.join(model_root, "running_setting.txt"), "w") as f:
                    f.write(record_setting)

            # ------------------- Set Models-Save Path ----------------------

            model_dir = os.path.join(model_root, "checkpoint")
            create_nonexistent_folder(model_dir)

            # ------------------- Set Logs-Save Path ----------------------

            log_dir = os.path.join(model_root, "logs")
            create_nonexistent_folder(log_dir)

            if pre_train_model_path:
                print("Loading Pre-trained Model...")
                model = load_model(multi_gpus, model, pre_train_model_path)

            # ------------- Assign Model to GPUS ----------------------

            if multi_gpus:
                print("DataParallel...")
                model = nn.DataParallel(model)

        # ------------------- Set Summary Writer ----------------------
        if tensorboard:
            self.summary_writer = SummaryWriter(logdir=log_dir)
        else:
            self.summary_writer = None

        # --------------- Training --------------

        save_top_1_model_flag = False
        if model.train_loader_type == "dali":
            self.train_loader_loop_func = self._dali_step
        elif model.train_loader_type == "torchvision":
            self.train_loader_loop_func = self._torchvision_step
        else:
            raise ValueError("The loader_loop_func of Module must be defined!")

        for epoch in range(self.start_epoch, train_epochs):
            if self.summary_writer:
                self.summary_writer.add_scalar(r"learning_rate", self.optimizer.param_groups[0]['lr'], epoch)

            print("\nTraining Epoch: %d/%d" % (epoch, train_epochs - 1))
            model.train()

            start_time_epoch = time.time()
            with tqdm(total=len(self.train_loader)) as pbar:
                for data_pairs in self.train_loader:
                    self.train_loader_loop_func(pbar, data_pairs)

            # Update Learning Rate
            self.lr_schedule.step()

            end_time_epoch = time.time()
            train_time_epoch = end_time_epoch - start_time_epoch
            if self.summary_writer:
                self.summary_writer.add_scalar("epoch_train_time", train_time_epoch, epoch)


        if self.summary_writer:
            self.summary_writer.close()
        print("Run Over!")

    def test(self):
        pass

    def _dali_step(self, pbar, data_pairs):
        for data_pair in data_pairs:
            data = data_pair["data"].to(self.device, non_blocking=True)
            target = data_pair["label"].squeeze().long().to(self.device, non_blocking=True).view(-1)

            self.optimizer.zero_grad()

            loss = self.step_func(self.model, data, target)

            loss.backward()
            self.optimizer.step()

            pbar.update(1)

    def _torchvision_step(self, pbar, data_pairs):
        data = data_pairs[0].to(self.device, non_blocking=True)
        target = data_pairs[1].squeeze().long().to(self.device, non_blocking=True).view(-1)

        self.optimizer.zero_grad()

        loss = self.step_func(self.model, data, target)

        loss.backward()
        self.optimizer.step()

        pbar.update(1)
