# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020/5/21 01:05

import os

import time
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from tqdm import tqdm

from . import CherryModule

from .utils.util import load_model, ContinualTrain, create_nonexistent_folder


class Runner(object):
    def __init__(self, seed=233, no_cuda=False):
        self.use_cuda = not no_cuda and torch.cuda.is_available()
        self.gpu_num = torch.cuda.device_count()
        self.multi_gpus = self.gpu_num > 1
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

    def fit(self, model: CherryModule, train_loader=None, optimizer=None, lr_schedule=None, train_step_func=None,
            save_path="./save", tensorboard=True, train_callbacks=None, val_callbacks=None, val_loader=None,
            continual_train_model_dir=None, record_setting: str = None, pre_train_model_path=None, train_epochs=0,
            checkpoint_callbacks=None, val_step_func=None):

        self.model = model
        self.model.to(self.device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = model.tc_optimizer()

        if lr_schedule:
            self.lr_schedule = lr_schedule(self.optimizer)
        else:
            self.lr_schedule = model.tc_lr_schedule(self.optimizer)

        if train_step_func:
            self.train_step_func = train_step_func
        else:
            self.train_step_func = model.tc_train_step

        if val_step_func:
            self.val_step_func = val_step_func
        else:
            self.val_step_func = model.tc_val_step

        if train_loader:
            self.train_loader = train_loader()
        else:
            self.train_loader = model.tc_train_loader()

        if val_loader:
            self.val_loader = val_loader()
        else:
            self.val_loader = model.tc_val_loader()

        self.save_path = save_path

        # ----------------- Continual Training -------------------------
        # Load Model, Optimizer
        # Set model_dir, log_dir, and start_epoch

        self.val_best_top_1 = - math.inf
        self.train_start_epoch = 0

        if continual_train_model_dir:
            continual_info = ContinualTrain.read(continual_train_model_dir)
            self.model_dir, self.log_dir, model_path, optimizer_path, lr_schedule_path, best_top_1 = continual_info

            self.optimizer.load_state_dict(torch.load(optimizer_path))
            self.lr_schedule.load_state_dict(torch.load(lr_schedule_path))

            print("Loading the Model of Epoch %d..." % self.lr_schedule.last_epoch)
            self.model = load_model(self.multi_gpus, self.model, model_path)
            self.train_start_epoch = self.lr_schedule.last_epoch
            self.val_best_top_1 = best_top_1

        else:
            # ------------------- Set Model-Save Path ----------------------

            self.model_root = self.save_path
            if not os.path.exists(self.model_root):
                os.makedirs(self.model_root)

            # ------------------- Save Running Setting ----------------------
            if record_setting:
                with open(os.path.join(self.model_root, "running_setting.txt"), "w") as f:
                    f.write(record_setting)

            # ------------------- Set Models-Save Path ----------------------

            self.model_dir = os.path.join(self.model_root, "checkpoint")
            create_nonexistent_folder(self.model_dir)

            # ------------------- Set Logs-Save Path ----------------------

            self.log_dir = os.path.join(self.model_root, "logs")
            create_nonexistent_folder(self.log_dir)

            if pre_train_model_path:
                print("Loading Pre-trained Model...")
                model = load_model(self.multi_gpus, self.model, pre_train_model_path)

            # ------------- Assign Model to GPUS ----------------------

            if self.multi_gpus and self.use_cuda:
                print("DataParallel...")
                model = nn.DataParallel(model)

        # ------------------- Set Summary Writer ----------------------
        if tensorboard:
            self.summary_writer = SummaryWriter(logdir=self.log_dir)
        else:
            self.summary_writer = None

        # --------------- Training --------------

        if model.train_loader_type == "dali":
            self.train_loader_loop_func = self._dali_step
        elif model.train_loader_type == "torchvision":
            self.train_loader_loop_func = self._torchvision_step
        else:
            raise ValueError("The loader_loop_func of Module must be defined!")

        for epoch in range(self.train_start_epoch, train_epochs):
            if self.summary_writer:
                self.summary_writer.add_scalar(r"learning_rate", self.optimizer.param_groups[0]['lr'], epoch)

            print("\nTraining Epoch: %d/%d" % (epoch, train_epochs - 1), "| Best Top-1:", self.val_best_top_1)
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

            # Test Training Dataset
            if train_callbacks:
                # training_metrics = [MetricAccuracy(1), MetricLoss(criterion)]
                print("Evaluating Training Data...")
                if model.train_loader_type == "dali":
                    training_metrics = self.eval_model_by_dali(self.train_loader, train_callbacks)
                elif model.train_loader_type == "torchvision":
                    training_metrics = self.eval_model_by_torch(self.train_loader, train_callbacks)
                else:
                    raise ValueError("The loader_loop_func of Module must be defined!")

                res_training = []
                for training_metric in training_metrics:
                    res_training.append(training_metric.get_metric())

                # save logs
                if self.summary_writer:
                    for train_metric in res_training:
                        self.summary_writer.add_scalar(r"train/" + train_metric["metric_name"], train_metric["metric_value"],
                                                  epoch)

            # Test Testing Dataset
            if val_callbacks:
                # valing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss(criterion)]
                print("Evaluating Valing Data...")
                if model.val_loader_type == "dali":
                    valing_metrics = self.eval_model_by_dali(self.val_loader, val_callbacks)
                elif model.val_loader_type == "torchvision":
                    valing_metrics = self.eval_model_by_torch(self.val_loader, val_callbacks)
                else:
                    raise ValueError("The loader_loop_func of Module must be defined!")

                res_val = []
                for metric in valing_metrics:
                    res_val.append(metric.get_metric())

                for val_metric in res_val:
                    if self.summary_writer:
                        self.summary_writer.add_scalar(r"val/" + val_metric["metric_name"], val_metric["metric_value"], epoch)

                    # Update Test Top_1
                    if val_metric["metric_name"] == "top_1":
                        if val_metric["metric_value"] > self.val_best_top_1:
                            print("Accuracy(%f) of Epoch %d is best now." % (val_metric["metric_value"], epoch))

                            self.val_best_top_1 = val_metric["metric_value"]
                            if self.summary_writer:
                                self.summary_writer.add_scalar(r"val_best_top1_acc", self.val_best_top_1, epoch)

            # save checkpoint
            if checkpoint_callbacks:
                for checkpoint_callback in checkpoint_callbacks:
                    checkpoint_callback.check2save(self, epoch)

            # flush Tensorboard
            if self.summary_writer:
                if epoch % 3 == 0:
                    self.summary_writer.flush()

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

            loss = self.train_step_func(self.model, data, target)

            loss.backward()
            self.optimizer.step()

            pbar.update(1)

    def _torchvision_step(self, pbar, data_pairs):
        data = data_pairs[0].to(self.device, non_blocking=True)
        target = data_pairs[1].squeeze().long().to(self.device, non_blocking=True).view(-1)

        self.optimizer.zero_grad()

        loss = self.train_step_func(self.model, data, target)

        loss.backward()
        self.optimizer.step()

        pbar.update(1)

    def eval_model_by_dali(self, loader, metrics):

        metrics_ = copy.deepcopy(metrics)

        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for data_pairs in loader:
                    for data_pair in data_pairs:
                        data = data_pair["data"].to(self.device, non_blocking=True)
                        target = data_pair["label"].squeeze().long().to(self.device, non_blocking=True).view(-1)
                        output_logits, loss = self.val_step_func(self.model, data, target)

                        for metric in metrics_:
                            metric.add_metric_record(output_logits, target, loss)

                        pbar.update(1)

        return metrics_

    def eval_model_by_torch(self, loader, metrics):
        metrics_ = copy.deepcopy(metrics)

        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for data_pair in loader:
                    data = data_pair[0].to(self.device, non_blocking=True)
                    target = data_pair[1].squeeze().long().to(self.device, non_blocking=True).view(-1)
                    output_logits, loss = self.val_step_func(self.model, data, target)

                    for metric in metrics_:
                        metric.add_metric_record(output_logits, target, loss)

                    pbar.update(1)

        return metrics_
