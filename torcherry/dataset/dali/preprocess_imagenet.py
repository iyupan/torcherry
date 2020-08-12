# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2019/10/23 14:14


import os

import math

# import torch.utils.data
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator


class Len_DALIClassificationIterator(DALIClassificationIterator):
    def __len__(self):
        return math.ceil(self._size / self.batch_size)


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, seed, shard_id,
                 decoder_type, cache_size, cache_threshold, cache_type, device_id=0, dali_cpu=False, gpu_num=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        if decoder_type == "cached":
            cache_size = cache_size
            cache_threshold = cache_threshold
            cache_type = cache_type
            print('Using nvJPEG with cache (size : {} threshold: {}, type: {})'
                  .format(cache_size, cache_threshold, cache_type))

            self.input = ops.FileReader(file_root=data_dir, shard_id=shard_id, num_shards=gpu_num, random_shuffle=True,
                                        stick_to_shard=True)
            self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB,
                                                cache_size=cache_size, cache_threshold=cache_threshold,
                                                cache_type=cache_type, cache_debug=False)
        else:
            print('Using nvJPEG')

            self.input = ops.FileReader(file_root=data_dir, shard_id=shard_id, num_shards=gpu_num, random_shuffle=True,
                                        stick_to_shard=False)
            self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)


        self.res = ops.RandomResizedCrop(device=dali_device, size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

        print('DALI "{0}" variant For Training.'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, size, seed, shard_id,
                 decoder_type, cache_size, cache_threshold, cache_type, device_id=0, dali_cpu=False,
                 gpu_num=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'


        if decoder_type == "cached":
            cache_size = cache_size
            cache_threshold = cache_threshold
            cache_type = cache_type
            print('Using nvJPEG with cache (size : {} threshold: {}, type: {})'
                  .format(cache_size, cache_threshold, cache_type))

            self.input = ops.FileReader(file_root=data_dir, shard_id=shard_id, num_shards=gpu_num,
                                        random_shuffle=False, stick_to_shard=True)
            self.decode = ops.ImageDecoder(device = decoder_device, output_type = types.RGB,
                                                cache_size=cache_size, cache_threshold=cache_threshold,
                                                cache_type=cache_type, cache_debug=False)
        else:
            print('Using nvJPEG')

            self.input = ops.FileReader(file_root=data_dir, shard_id=shard_id, num_shards=gpu_num,
                                        random_shuffle=False, stick_to_shard=False)
            self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)

        self.res = ops.Resize(device=dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        print('DALI "{0}" variant For Testing.'.format(dali_device))

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, gpu_num, crop, seed,
                           decoder_type="", cache_size=0, cache_threshold=0, cache_type="none", val_size=256,
                           dali_cpu=False, auto_reset=True):
    if type == 'train':
        dali_pipes = []
        for shard_id in range(gpu_num):
            pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, shard_id=shard_id,
                                        data_dir=os.path.join(image_dir, 'train'), seed=seed, dali_cpu=dali_cpu,
                                        crop=crop, gpu_num=gpu_num, decoder_type=decoder_type, cache_size=cache_size,
                                        cache_threshold=cache_threshold, cache_type=cache_type)
            pip_train.build()
            dali_pipes.append(pip_train)

        dali_trains = Len_DALIClassificationIterator(dali_pipes,
                                                             fill_last_batch=False,
                                                             auto_reset=auto_reset,
                                                     reader_name="Reader"
                                                             )
        return dali_trains
    elif type == 'val':
        dali_pipes = []
        for shard_id in range(gpu_num):
            pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, shard_id=shard_id,
                                    data_dir=os.path.join(image_dir, 'val'), seed=seed, dali_cpu=dali_cpu,
                                    crop=crop, size=val_size, gpu_num=gpu_num, decoder_type=decoder_type,
                                    cache_size=cache_size, cache_threshold=cache_threshold, cache_type=cache_type)
            pip_val.build()
            dali_pipes.append(pip_val)

        dali_vals = Len_DALIClassificationIterator(dali_pipes,
                                                     fill_last_batch=False,
                                                     auto_reset=auto_reset,
                                                   reader_name="Reader"
                                                     )
        return dali_vals


# def get_imagenet_iter_torch(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
#                             world_size=1, local_rank=0):
#     if type == 'train':
#         transform = transforms.Compose([
#             transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         dataset = datasets.ImageFolder(os.path.join(image_dir, 'train'), transform)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
#                                                  pin_memory=True)
#     elif type == 'val':
#         transform = transforms.Compose([
#             transforms.Resize(val_size),
#             transforms.CenterCrop(crop),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         dataset = datasets.ImageFolder(os.path.join(image_dir, 'val'), transform)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads,
#                                                  pin_memory=True)
#     else:
#         raise  ValueError("The type %s is not founded." % type)
#
#     return dataloader
