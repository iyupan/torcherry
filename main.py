# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020/5/21 00:36

from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(1)

import os
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

# from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
# from torchvision import transforms

from torcherry.dataset.dali import get_mnist_iter_dali, get_cifar10_iter_dali

import torcherry as tc
from torcherry.utils.metric import MetricAccuracy, MetricLoss
from torcherry.utils.checkpoint import CheckBestValAcc, CheckContinueTrain, CheckFrequence
from torcherry.utils.util import set_env_seed


class CModel(tc.CherryModule):
    def __init__(self):
        super().__init__()
        self.gpu_num = torch.cuda.device_count()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def tc_train_step(self, model, data, target):
        output_logits = model(data)
        loss = F.cross_entropy(output_logits, target)
        return loss

    def tc_val_step(self, model, data, target):
        output_logits = model(data)
        loss = F.cross_entropy(output_logits, target)
        return output_logits, loss

    def tc_test_step(self, model, data, target):
        output_logits = model(data)
        loss = F.cross_entropy(output_logits, target)
        return output_logits, loss

    def tc_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def tc_lr_schedule(self, optimizer):
        return MultiStepLR(optimizer, [90, 130, 170], 0.2)

    def tc_train_loader(self):
        # self.train_loader_type = "torchvision"
        # dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        # loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        # return loader
        self.train_loader_type = "dali"
        return get_mnist_iter_dali(type='train', image_dir=os.getcwd(), batch_size=31, dali_cpu=True,
                                   num_threads=4, seed=233, gpu_num=self.gpu_num)
        # return get_cifar10_iter_dali(type='train', image_dir=os.getcwd(), batch_size=32, dali_cpu=False,
        #                            num_threads=4, seed=233, gpu_num=1)

    def tc_val_loader(self):
        # self.val_loader_type = "torchvision"
        # dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        # loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        # return loader
        self.val_loader_type = "dali"
        return get_mnist_iter_dali(type='val', image_dir=os.getcwd(), batch_size=31, dali_cpu=False,
                                   num_threads=4, seed=233, gpu_num=self.gpu_num)
        # return get_cifar10_iter_dali(type='val', image_dir=os.getcwd(), batch_size=32, dali_cpu=False,
        #                            num_threads=4, seed=233, gpu_num=1)

    def tc_test_loader(self):
        # self.test_loader_type = "torchvision"
        # dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        # loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        # return loader
        self.test_loader_type = "dali"
        return get_mnist_iter_dali(type='val', image_dir=os.getcwd(), batch_size=31, dali_cpu=False,
                                   num_threads=4, seed=233, gpu_num=1)


if __name__ == '__main__':
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    set_env_seed(233, use_cuda)

    save_path = os.path.join("./save", "tt", time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))

    model = CModel()

    training_metrics = [MetricAccuracy(1), MetricLoss()]
    valing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    check_metrics = [CheckBestValAcc(), CheckContinueTrain(3)]

    runner = tc.Runner(use_cuda)
    res_dict = runner.fit(model, save_path=save_path, train_epochs=20, train_callbacks=training_metrics,
               val_callbacks=valing_metrics, checkpoint_callbacks=check_metrics,
               )

    # testing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    # runner.test(model, "/media/windows_e/Ubuntu_Project/Github/torcherry/save/tt/20200521_201401/checkpoint/model-nn-best-val-top-1.pt", test_callbacks=testing_metrics)
    pass
