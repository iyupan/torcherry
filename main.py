# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020/5/21 00:36

from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(1, 90)

import os
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import torcherry as tc


class CModel(tc.CherryModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def tc_train_step(self, model, data, target):
        y_hat = model(data)
        loss = F.cross_entropy(y_hat, target)
        return loss

    def tc_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def tc_train_loader(self):
        self.train_loader_type = "torchvision"
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        return loader

    def tc_lr_schedule(self, optimizer):
        return MultiStepLR(optimizer, [90, 130, 170], 0.2)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = CModel()

    save_path = os.path.join("./save" "tt", time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))

    runner = tc.Runner()
    runner.fit(model, device, save_path=save_path, train_epochs=20)
    pass
