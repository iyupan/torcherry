# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020-02-10 13:30


import os
import sys
import time

import math
from distutils.version import StrictVersion

import nvidia.dali
assert StrictVersion(nvidia.dali.__version__) >= StrictVersion("1.0.0"), "Dali version should be higher than 1.0.0!"

import nvidia.dali.tfrecord as tfrec
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

from .util_cifar100 import process_cifar100


class Len_DALIClassificationIterator(DALIClassificationIterator):
    def __len__(self):
        return math.ceil(self._size / self.batch_size)


@pipeline_def
def create_dali_cifar100_tfrec_pipeline(data_path, data_index_path, type, num_shards, shard_id, crop=28, dali_cpu=False):
    dali_device = 'cpu' if dali_cpu else 'gpu'

    if type == "train":
        inputs = fn.readers.tfrecord(path=data_path,
                                        index_path=data_index_path,
                                        features={
                                            'image': tfrec.FixedLenFeature((), tfrec.string, ""),
                                            'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                        },
                                        num_shards=num_shards,
                                        shard_id=shard_id,
                                        name="Reader",
                                        random_shuffle=True,
                                        initial_fill=2048,
                                        )
        images = inputs["image"]
        images = fn.reshape(images.gpu(),
                            device=dali_device,
                            shape=[32, 32, 3],
                            layout="HWC")
        images = fn.paste(images,
                          device=dali_device,
                          ratio=1.25,
                          fill_value=0)
        uniform = fn.random.uniform(range=(0., 1.))
        images = fn.crop(images,
                         device=dali_device,
                         crop_h=crop,
                         crop_w=crop,
                         crop_pos_x=uniform,
                         crop_pos_y=uniform)
        images = fn.crop_mirror_normalize(images,
                                          device=dali_device,
                                          dtype=types.FLOAT,
                                          output_layout=types.NCHW,
                                          mean=[0.5070751592371323 * 255., 0.48654887331495095 * 255.,
                                                0.4409178433670343 * 255.],
                                          std=[0.2673342858792401 * 255., 0.2564384629170883 * 255.,
                                                0.27615047132568404 * 255.],
                                          mirror=fn.random.coin_flip(probability=0.5))
        labels = inputs["label"].gpu()

    elif type == "val":
        inputs = fn.readers.tfrecord(path=data_path,
                                     index_path=data_index_path,
                                     features={
                                         'image': tfrec.FixedLenFeature((), tfrec.string, ""),
                                         'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                                     },
                                     num_shards=num_shards,
                                     shard_id=shard_id,
                                     name="Reader",
                                     )
        images = inputs["image"]
        images = fn.reshape(images.gpu(),
                            device=dali_device,
                            shape=[32, 32, 3],
                            layout="HWC")
        images = fn.crop_mirror_normalize(images,
                                          device=dali_device,
                                          dtype=types.FLOAT,
                                          output_layout=types.NCHW,
                                          mean=[0.5070751592371323 * 255., 0.48654887331495095 * 255.,
                                                0.4409178433670343 * 255.],
                                          std=[0.2673342858792401 * 255., 0.2564384629170883 * 255.,
                                                0.27615047132568404 * 255.],
                                          mirror=False)
        labels = inputs["label"].gpu()
    else:
        raise ValueError("Type %s is not existed!" % type)

    return images, labels


def get_cifar100_iter_dali(type, image_dir, batch_size, num_threads, seed, dali_cpu, gpu_num, auto_reset=True):
    process_cifar100(image_dir)
    data_path = os.path.join(image_dir, "CIFAR100_tfrecords", "processed")

    if type == 'train':
        train_image_path=os.path.join(data_path, "cifar100_train.tfrecords")
        train_image_index_path=os.path.join(data_path, "cifar100_train_idx")

        pipes = []
        for i in range(gpu_num):
            pipe = create_dali_cifar100_tfrec_pipeline(data_path=train_image_path,
                                                    data_index_path=train_image_index_path,
                                                    type="train",

                                                    batch_size=batch_size,
                                                    num_threads=num_threads,
                                                    device_id=0,
                                                    seed=seed,
                                                    crop=32,
                                                    dali_cpu=dali_cpu,
                                                    shard_id=i,
                                                    num_shards=gpu_num,
                                                    )
            pipe.build()
            pipes.append(pipe)

    elif type == 'val':
        val_image_path = os.path.join(data_path, "cifar100_test.tfrecords")
        val_image_index_path = os.path.join(data_path, "cifar100_test_idx")

        pipes = []
        for i in range(gpu_num):
            pipe = create_dali_cifar100_tfrec_pipeline(data_path=val_image_path,
                                                    data_index_path=val_image_index_path,
                                                    type="val",

                                                    batch_size=batch_size,
                                                    num_threads=num_threads,
                                                    device_id=0,
                                                    seed=seed,
                                                    crop=32,
                                                    dali_cpu=dali_cpu,
                                                    shard_id=i,
                                                    num_shards=gpu_num,
                                                    )
            pipe.build()
            pipes.append(pipe)
    else:
        raise ValueError("Type %s is not existed!" % type)

    data_loader = Len_DALIClassificationIterator(pipes, reader_name="Reader", auto_reset=auto_reset,
                                                 last_batch_policy=LastBatchPolicy.PARTIAL)
    return data_loader

# class HybridTrainPipe_CIFAR100(Pipeline):
#     def __init__(self, batch_size, num_threads, data_dir, seed, gpu_num, shard_id, crop=32, device_id=0, dali_cpu=False):
#         super(HybridTrainPipe_CIFAR100, self).__init__(batch_size, num_threads, device_id, seed=seed)
#         dali_device = 'cpu' if dali_cpu else 'gpu'
#         data_path = os.path.join(data_dir, "CIFAR100_tfrecords", "processed")
#
#         self.input = ops.TFRecordReader(path=os.path.join(data_path, "cifar100_train.tfrecords"),
#                                         index_path=os.path.join(data_path, "cifar100_train_idx"),
#                                         features={
#                                             'image': tfrec.FixedLenFeature((), tfrec.string, ""),
#                                             'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
#                                         },
#                                         num_shards=gpu_num,
#                                         shard_id=shard_id,
#                                         random_shuffle=True,
#                                         initial_fill=2048,
#                                         )
#
#         self.reshape = ops.Reshape(device=dali_device, shape=[32, 32, 3], layout="HWC")
#
#         self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
#         self.uniform = ops.Uniform(range=(0., 1.))
#         self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
#         self.cmnp = ops.CropMirrorNormalize(device=dali_device,
#                                             dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             mean=[0.5070751592371323 * 255., 0.48654887331495095 * 255.,
#                                                   0.4409178433670343 * 255.],
#                                             std=[0.2673342858792401 * 255., 0.2564384629170883 * 255.,
#                                                  0.27615047132568404 * 255.]
#
#                                             )
#         self.coin = ops.CoinFlip(probability=0.5)
#
#     def define_graph(self):
#         rng = self.coin()
#
#         inputs = self.input(name="Reader")
#         output = inputs["image"].gpu()
#         output = self.reshape(output)
#         output = self.pad(output)
#         output = self.crop(output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
#         output = self.cmnp(output, mirror=rng)
#
#         return (output, inputs["label"].gpu())
#
#     def iter_setup(self):
#         pass
#
#
# class HybridValPipe_CIFAR100(Pipeline):
#     def __init__(self, batch_size, num_threads, data_dir, dali_cpu, seed, gpu_num, shard_id, device_id=0):
#         super(HybridValPipe_CIFAR100, self).__init__(batch_size, num_threads, device_id, seed=seed)
#         dali_device = 'cpu' if dali_cpu else 'gpu'
#         data_path = os.path.join(data_dir, "CIFAR100_tfrecords", "processed")
#
#         self.input = ops.TFRecordReader(path=os.path.join(data_path, "cifar100_test.tfrecords"),
#                                         index_path=os.path.join(data_path, "cifar100_test_idx"),
#                                         features={
#                                             'image': tfrec.FixedLenFeature((), tfrec.string, ""),
#                                             'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
#                                         },
#                                         num_shards=gpu_num,
#                                         shard_id=shard_id,
#                                         )
#
#         self.reshape = ops.Reshape(device=dali_device, shape=[32, 32, 3], layout="HWC")
#         self.cmnp = ops.CropMirrorNormalize(device=dali_device,
#                                             dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             mean=[0.5070751592371323 * 255., 0.48654887331495095 * 255.,
#                                                   0.4409178433670343 * 255.],
#                                             std=[0.2673342858792401 * 255., 0.2564384629170883 * 255.,
#                                                  0.27615047132568404 * 255.]
#                                             )
#
#     def define_graph(self):
#         inputs = self.input(name="Reader")
#         output = inputs["image"].gpu()
#         output = self.reshape(output)
#         output = self.cmnp(output)
#         return (output, inputs["label"].gpu())
#
#     def iter_setup(self):
#         pass
#
#
#
# def get_cifar100_iter_dali(type, image_dir, batch_size, num_threads, seed, dali_cpu, gpu_num, auto_reset=True):
#     process_cifar100(image_dir)
#
#     if type == 'train':
#
#         pipes = []
#         for i in range(gpu_num):
#
#             pip_train = HybridTrainPipe_CIFAR100(batch_size=batch_size, num_threads=num_threads, shard_id=i,
#                                               gpu_num=gpu_num, data_dir=image_dir, seed=seed, dali_cpu=dali_cpu,
#                                               crop=32)
#             pip_train.build()
#             pipes.append(pip_train)
#         dali_iter_train = Len_DALIClassificationIterator(pipes,
#                                                      fill_last_batch=True, auto_reset=auto_reset, reader_name="Reader")
#         return dali_iter_train
#
#     elif type == 'val':
#         pipes = []
#         for i in range(gpu_num):
#             pip_val = HybridValPipe_CIFAR100(batch_size=batch_size, num_threads=num_threads, shard_id=i,
#                                           gpu_num=gpu_num, data_dir=image_dir, seed=seed, dali_cpu=dali_cpu)
#             pip_val.build()
#             pipes.append(pip_val)
#         dali_iter_val = Len_DALIClassificationIterator(pipes,
#                                                    fill_last_batch=False, auto_reset=auto_reset, reader_name="Reader")
#         return dali_iter_val
