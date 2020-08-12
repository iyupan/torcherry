# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020-02-10 13:30

import os
import sys
import time

import math

# import torch
# import torchvision.transforms as transforms
# from torchvision.datasets import SVHN

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.pipeline import Pipeline

from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from .util_svhn import process_svhn


class Len_DALIClassificationIterator(DALIClassificationIterator):
    def __len__(self):
        return math.ceil(self._size / self.batch_size)


class HybridTrainPipe_SVHN(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, seed, gpu_num, shard_id, crop=32, device_id=0, dali_cpu=False):
        super(HybridTrainPipe_SVHN, self).__init__(batch_size, num_threads, device_id, seed=seed)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        data_path = os.path.join(data_dir, "SVHN_tfrecords", "processed")

        self.input = ops.TFRecordReader(path=[os.path.join(data_path, "svhn_train.tfrecords"),
                                              os.path.join(data_path, "svhn_extra.tfrecords")],
                                        index_path=[os.path.join(data_path, "svhn_train_idx"),
                                                    os.path.join(data_path, "svhn_extra_idx")],
                                        features={
                                            'image': tfrec.FixedLenFeature((), tfrec.string, ""),
                                            'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                        },
                                        num_shards=gpu_num,
                                        shard_id=shard_id,
                                        random_shuffle=True,
                                        initial_fill=2048,
                                        )

        self.reshape = ops.Reshape(device=dali_device, shape=[32, 32, 3], layout="HWC")

        # self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        # self.uniform = ops.Uniform(range=(0., 1.))
        # self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0.5 * 255., 0.5 * 255.,
                                                  0.5 * 255.],
                                            std=[0.5 * 255., 0.5 * 255.,
                                                 0.5 * 255.]

                                            )
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()

        inputs = self.input(name="Reader")
        output = inputs["image"].gpu()
        output = self.reshape(output)
        # output = self.pad(output)
        # output = self.crop(output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.cmnp(output, mirror=rng)

        return (output, inputs["label"].gpu())

    def iter_setup(self):
        pass


class HybridValPipe_SVHN(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, dali_cpu, seed, gpu_num, shard_id, device_id=0):
        super(HybridValPipe_SVHN, self).__init__(batch_size, num_threads, device_id, seed=seed)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        data_path = os.path.join(data_dir, "SVHN_tfrecords", "processed")

        self.input = ops.TFRecordReader(path=os.path.join(data_path, "svhn_test.tfrecords"),
                                        index_path=os.path.join(data_path, "svhn_test_idx"),
                                        features={
                                            'image': tfrec.FixedLenFeature((), tfrec.string, ""),
                                            'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                        },
                                        num_shards=gpu_num,
                                        shard_id=shard_id,
                                        )

        self.reshape = ops.Reshape(device=dali_device, shape=[32, 32, 3], layout="HWC")
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0.5 * 255., 0.5 * 255.,
                                                  0.5 * 255.],
                                            std=[0.5 * 255., 0.5 * 255.,
                                                 0.5 * 255.]
                                            )

    def define_graph(self):
        inputs = self.input(name="Reader")
        output = inputs["image"].gpu()
        output = self.reshape(output)
        output = self.cmnp(output)
        return (output, inputs["label"].gpu())

    def iter_setup(self):
        pass



def get_svhn_iter_dali(type, image_dir, batch_size, num_threads, seed, dali_cpu, gpu_num, auto_reset=True):
    process_svhn(image_dir)

    if type == 'train':
        pipes = []
        for i in range(gpu_num):

            pip_train = HybridTrainPipe_SVHN(batch_size=batch_size, num_threads=num_threads, shard_id=i,
                                              gpu_num=gpu_num, data_dir=image_dir, seed=seed, dali_cpu=dali_cpu,
                                              crop=32)
            pip_train.build()
            pipes.append(pip_train)

        dali_iter_train = Len_DALIClassificationIterator(pipes,
                                                     fill_last_batch=True, auto_reset=auto_reset, reader_name="Reader")
        return dali_iter_train

    elif type == 'val':
        pipes = []
        for i in range(gpu_num):
            pip_val = HybridValPipe_SVHN(batch_size=batch_size, num_threads=num_threads, shard_id=i,
                                          gpu_num=gpu_num, data_dir=image_dir, seed=seed, dali_cpu=dali_cpu)
            pip_val.build()
            pipes.append(pip_val)

        iter_size = pipes[0].epoch_size("Reader")
        dali_iter_val = Len_DALIClassificationIterator(pipes,
                                                   fill_last_batch=False, auto_reset=auto_reset, reader_name="Reader")
        return dali_iter_val


# def get_cifar_iter_torch(type, image_dir, batch_size, num_threads, cutout=0):
#     CIFAR_MEAN = [0.5, 0.5, 0.5]
#     CIFAR_STD = [0.5, 0.5, 0.5]
#     if type == 'train':
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#         ])
#         train_dst = SVHN(root=image_dir, split="train", download=True, transform=transform_train)
#         train_iter = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, pin_memory=True,
#                                                  num_workers=num_threads)
#         return train_iter
#     elif type == "test":
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#         ])
#         test_dst = SVHN(root=image_dir, split="test", download=True, transform=transform_test)
#         test_iter = torch.utils.data.DataLoader(test_dst, batch_size=batch_size, shuffle=False, pin_memory=True,
#                                                 num_workers=num_threads)
#         return test_iter
#
#     else:
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#         ])
#         extra_dst = SVHN(root=image_dir, split="extra", download=True, transform=transform_test)
#         extra_iter = torch.utils.data.DataLoader(extra_dst, batch_size=batch_size, shuffle=False, pin_memory=True,
#                                                 num_workers=num_threads)
#         return extra_iter
#
# if __name__ == '__main__':
#     train_loader = get_svhn_iter_dali(type='train', image_dir='/home/panyu/download/datas', batch_size=256,
#                                        num_threads=4, seed=233, dali_cpu=True)
#     print('start iterate')
#     start = time.time()
#     for i, data in enumerate(train_loader):
#         images = data[0]["data"].cuda(non_blocking=True)
#         labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
#     end = time.time()
#     print('end iterate')
#     print('dali iterate time: %fs' % (end - start))
#
#     train_loader = get_cifar_iter_torch(type='train', image_dir='/home/panyu/download/datas', batch_size=256,
#                                         num_threads=4)
#     print('start iterate')
#     start = time.time()
#     for i, data in enumerate(train_loader):
#         images = data[0].cuda(non_blocking=True)
#         labels = data[1].cuda(non_blocking=True)
#     end = time.time()
#     print('end iterate')
#     print('torch iterate time: %fs' % (end - start))
