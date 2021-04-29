# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020-02-09 23:28

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

from .util_mnist import process_mnist


class Len_DALIClassificationIterator(DALIClassificationIterator):
    def __len__(self):
        return math.ceil(self._size / self.batch_size)


# @pipeline_def
# def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
#     images, labels = fn.readers.file(file_root=data_dir,
#                                      shard_id=shard_id,
#                                      num_shards=num_shards,
#                                      random_shuffle=is_training,
#                                      pad_last_batch=True,
#                                      name="Reader")
#     dali_device = 'cpu' if dali_cpu else 'gpu'
#     decoder_device = 'cpu' if dali_cpu else 'mixed'
#     device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
#     host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
#     if is_training:
#         images = fn.decoders.image_random_crop(images,
#                                                device=decoder_device, output_type=types.RGB,
#                                                device_memory_padding=device_memory_padding,
#                                                host_memory_padding=host_memory_padding,
#                                                random_aspect_ratio=[0.8, 1.25],
#                                                random_area=[0.1, 1.0],
#                                                num_attempts=100)
#         images = fn.resize(images,
#                            device=dali_device,
#                            resize_x=crop,
#                            resize_y=crop,
#                            interp_type=types.INTERP_TRIANGULAR)
#         mirror = fn.random.coin_flip(probability=0.5)
#     else:
#         images = fn.decoders.image(images,
#                                    device=decoder_device,
#                                    output_type=types.RGB)
#         images = fn.resize(images,
#                            device=dali_device,
#                            size=size,
#                            mode="not_smaller",
#                            interp_type=types.INTERP_TRIANGULAR)
#         mirror = False
#
#     images = fn.crop_mirror_normalize(images.gpu(),
#                                       dtype=types.FLOAT,
#                                       output_layout="CHW",
#                                       crop=(crop, crop),
#                                       mean=[0.485 * 255,0.456 * 255,0.406 * 255],
#                                       std=[0.229 * 255,0.224 * 255,0.225 * 255],
#                                       mirror=mirror)
#     labels = labels.gpu()
#     return images, labels


@pipeline_def
def create_dali_mnist_tfrec_pipeline(data_path, data_index_path, type, num_shards, shard_id, crop=28, dali_cpu=False):
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
                            shape=[28, 28, 1],
                            layout="HWC")
        images = fn.crop_mirror_normalize(images,
                                          device=dali_device,
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          mean=[0.1307 * 255.],
                                          std=[0.3081 * 255.],
                                          mirror=False)
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
                            shape=[28, 28, 1],
                            layout="HWC")
        images = fn.crop_mirror_normalize(images,
                                          device=dali_device,
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          mean=[0.1307 * 255.],
                                          std=[0.3081 * 255.],
                                          mirror=False)
        labels = inputs["label"].gpu()
    else:
        raise ValueError("Type %s is not existed!" % type)

    return images, labels


def get_mnist_iter_dali(type, image_dir, batch_size, num_threads, seed, dali_cpu, gpu_num, auto_reset=True):
    process_mnist(image_dir)
    data_path = os.path.join(image_dir, "MNIST_tfrecords", "processed")

    if type == 'train':
        train_image_path=os.path.join(data_path, "mnist_train.tfrecords")
        train_image_index_path=os.path.join(data_path, "mnist_train_idx")

        pipes = []
        for i in range(gpu_num):
            pipe = create_dali_mnist_tfrec_pipeline(data_path=train_image_path,
                                                    data_index_path=train_image_index_path,
                                                    type="train",

                                                    batch_size=batch_size,
                                                    num_threads=num_threads,
                                                    device_id=0,
                                                    seed=seed,
                                                    crop=28,
                                                    dali_cpu=dali_cpu,
                                                    shard_id=i,
                                                    num_shards=gpu_num,
                                                    )
            pipe.build()
            pipes.append(pipe)

    elif type == 'val':
        val_image_path = os.path.join(data_path, "mnist_test.tfrecords")
        val_image_index_path = os.path.join(data_path, "mnist_test_idx")

        pipes = []
        for i in range(gpu_num):
            pipe = create_dali_mnist_tfrec_pipeline(data_path=val_image_path,
                                                    data_index_path=val_image_index_path,
                                                    type="val",

                                                    batch_size=batch_size,
                                                    num_threads=num_threads,
                                                    device_id=0,
                                                    seed=seed,
                                                    crop=28,
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


# class HybridTrainPipe_MNIST(Pipeline):
#     def __init__(self, batch_size, num_threads, data_dir, seed, gpu_num, shard_id, crop=28, device_id=0, dali_cpu=False):
#         super(HybridTrainPipe_MNIST, self).__init__(batch_size, num_threads, device_id, seed=seed)
#         dali_device = 'cpu' if dali_cpu else 'gpu'
#         data_path = os.path.join(data_dir, "MNIST_tfrecords", "processed")
#
#         self.input = fn.readers.tfrecord(path=os.path.join(data_path, "mnist_train.tfrecords"),
#                                         index_path=os.path.join(data_path, "mnist_train_idx"),
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
#         self.reshape = fn.Reshape(device=dali_device, shape=[28, 28, 1], layout="HWC")
#         self.cmnp = fn.CropMirrorNormalize(device=dali_device,
#                                             dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             mean=[0.1307 * 255.],
#                                             std=[0.3081 * 255.]
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


# class HybridValPipe_MNIST(Pipeline):
#     def __init__(self, batch_size, num_threads, data_dir, dali_cpu, seed, gpu_num, shard_id, device_id=0):
#         super(HybridValPipe_MNIST, self).__init__(batch_size, num_threads, device_id, seed=seed)
#         dali_device = 'cpu' if dali_cpu else 'gpu'
#         data_path = os.path.join(data_dir, "MNIST_tfrecords", "processed")
#
#         self.input = fn.readers.tfrecord(path=os.path.join(data_path, "mnist_test.tfrecords"),
#                                         index_path=os.path.join(data_path, "mnist_test_idx"),
#                                         features={
#                                             'image': tfrec.FixedLenFeature((), tfrec.string, ""),
#                                             'label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
#                                         },
#                                         num_shards=gpu_num,
#                                         shard_id=shard_id,
#                                         )
#
#         self.reshape = ops.Reshape(device=dali_device, shape=[28, 28, 1], layout="HWC")
#         self.cmnp = ops.CropMirrorNormalize(device=dali_device,
#                                             dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             mean=[0.1307 * 255.],
#                                             std=[0.3081 * 255.]
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


# def get_mnist_iter_dali(type, image_dir, batch_size, num_threads, seed, dali_cpu, gpu_num, auto_reset=True):
#     process_mnist(image_dir)
#
#     if type == 'train':
#
#         pipes = []
#         for i in range(gpu_num):
#
#             pip_train = HybridTrainPipe_MNIST(batch_size=batch_size, num_threads=num_threads, shard_id=i,
#                                               gpu_num=gpu_num, data_dir=image_dir, seed=seed, dali_cpu=dali_cpu,
#                                               crop=28)
#             pip_train.build()
#             pipes.append(pip_train)
#         dali_iter_train = Len_DALIClassificationIterator(pipes,
#                                                      fill_last_batch=True, auto_reset=auto_reset, reader_name="Reader")
#         return dali_iter_train
#
#     elif type == 'val':
#         pipes = []
#         for i in range(gpu_num):
#             pip_val = HybridValPipe_MNIST(batch_size=batch_size, num_threads=num_threads, shard_id=i,
#                                           gpu_num=gpu_num, data_dir=image_dir, seed=seed, dali_cpu=dali_cpu)
#             pip_val.build()
#             pipes.append(pip_val)
#         dali_iter_val = Len_DALIClassificationIterator(pipes,
#                                                    fill_last_batch=False, auto_reset=auto_reset, reader_name="Reader")
#         return dali_iter_val
