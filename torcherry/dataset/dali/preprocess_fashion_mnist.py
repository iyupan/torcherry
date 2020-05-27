# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020-02-09 23:28


# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2019/10/23 14:03


import os
import sys
import time

import math

import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.pipeline import Pipeline

from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from .util_fashion_mnist import process_fashion_mnist


class Len_DALIClassificationIterator(DALIClassificationIterator):
    def __len__(self):
        return math.ceil(self._size / self.batch_size)


class HybridTrainPipe_FASHION_MNIST(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, seed, gpu_num, shard_id, crop=28, device_id=0, dali_cpu=False):
        super(HybridTrainPipe_FASHION_MNIST, self).__init__(batch_size, num_threads, device_id, seed=seed)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        data_path = os.path.join(data_dir, "FASHION_MNIST_tfrecords", "processed")

        self.input = ops.TFRecordReader(path=os.path.join(data_path, "fashion_mnist_train.tfrecords"),
                                        index_path=os.path.join(data_path, "fashion_mnist_train_idx"),
                                        features={
                                            'image': tfrec.FixedLenFeature((), tfrec.string, ""),
                                            'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                        },
                                        num_shards=gpu_num,
                                        shard_id=shard_id,
                                        random_shuffle=True,
                                        initial_fill=2048,
                                        )

        self.reshape = ops.Reshape(device=dali_device, shape=[28, 28, 1], layout="HWC")
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.5 * 255.],
                                            std=[0.5 * 255.]
                                            )

    def define_graph(self):
        inputs = self.input(name="Reader")
        output = inputs["image"].gpu()
        output = self.reshape(output)
        output = self.cmnp(output)
        return (output, inputs["label"].gpu())

    def iter_setup(self):
        pass


class HybridValPipe_FASHION_MNIST(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, dali_cpu, seed, gpu_num, shard_id, device_id=0):
        super(HybridValPipe_FASHION_MNIST, self).__init__(batch_size, num_threads, device_id, seed=seed)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        data_path = os.path.join(data_dir, "FASHION_MNIST_tfrecords", "processed")

        self.input = ops.TFRecordReader(path=os.path.join(data_path, "fashion_mnist_test.tfrecords"),
                                        index_path=os.path.join(data_path, "fashion_mnist_test_idx"),
                                        features={
                                            'image': tfrec.FixedLenFeature((), tfrec.string, ""),
                                            'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                        },
                                        num_shards=gpu_num,
                                        shard_id=shard_id,
                                        )

        self.reshape = ops.Reshape(device=dali_device, shape=[28, 28, 1], layout="HWC")
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.5 * 255.],
                                            std=[0.5 * 255.]
                                            )

    def define_graph(self):
        inputs = self.input(name="Reader")
        output = inputs["image"].gpu()
        output = self.reshape(output)
        output = self.cmnp(output)
        return (output, inputs["label"].gpu())

    def iter_setup(self):
        pass



def get_fashion_mnist_iter_dali(type, image_dir, batch_size, num_threads, seed, dali_cpu, gpu_num, auto_reset=True):
    process_fashion_mnist(image_dir)

    if type == 'train':

        pipes = []
        for i in range(gpu_num):

            pip_train = HybridTrainPipe_FASHION_MNIST(batch_size=batch_size, num_threads=num_threads, shard_id=i,
                                              gpu_num=gpu_num, data_dir=image_dir, seed=seed, dali_cpu=dali_cpu,
                                              crop=28)
            pip_train.build()
            pipes.append(pip_train)
        iter_size = pipes[0].epoch_size("Reader")
        dali_iter_train = Len_DALIClassificationIterator(pipes, size=iter_size,
                                                     fill_last_batch=True, auto_reset=auto_reset)
        return dali_iter_train

    elif type == 'val':
        pipes = []
        for i in range(gpu_num):
            pip_val = HybridValPipe_FASHION_MNIST(batch_size=batch_size, num_threads=num_threads, shard_id=i,
                                          gpu_num=gpu_num, data_dir=image_dir, seed=seed, dali_cpu=dali_cpu)
            pip_val.build()
            pipes.append(pip_val)
        iter_size = pipes[0].epoch_size("Reader")
        dali_iter_val = Len_DALIClassificationIterator(pipes, size=iter_size,
                                                   fill_last_batch=False, auto_reset=auto_reset)
        return dali_iter_val


def get_fashion_mnist_iter_torch(type, image_dir, batch_size, num_threads, cutout=0):
    FASHION_MNIST_MEAN = [0.5,]
    FASHION_MNIST_STD = [0.5,]
    if type == 'train':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ])
        train_dst = FashionMNIST(root=image_dir, train=True, download=True, transform=transform_train)
        train_iter = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                 num_workers=num_threads)
        return train_iter
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ])
        test_dst = FashionMNIST(root=image_dir, train=False, download=True, transform=transform_test)
        test_iter = torch.utils.data.DataLoader(test_dst, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                num_workers=num_threads)
        return test_iter


if __name__ == '__main__':
    train_loader = get_fashion_mnist_iter_dali(type='train', image_dir='/home/panyu/download/datas', batch_size=256,
                                       num_threads=4, seed=233, dali_cpu=True)
    print('start iterate')
    start = time.time()
    for i, data in enumerate(train_loader):
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    end = time.time()
    print('end iterate')
    print('dali iterate time: %fs' % (end - start))

    train_loader = get_fashion_mnist_iter_torch(type='train', image_dir='/home/panyu/download/datas', batch_size=256,
                                        num_threads=4)
    print('start iterate')
    start = time.time()
    for i, data in enumerate(train_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    print('end iterate')
    print('torch iterate time: %fs' % (end - start))
