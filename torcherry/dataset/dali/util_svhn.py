# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020-02-10 13:10


import os
import errno

from torchvision.datasets.utils import download_and_extract_archive


_url = "https://storage.live.com/items/709703FB7C5DF5D6!124?.&authkey=!AOQdfXbdmzSWenU&e=edCa1V&ithint=.zip"

_MD5 = "64c3bfca2828edaeed5cdd3b8925e305"
_filename = "SVHN_records.zip"

_data_root_name = "SVHN_tfrecords"

_svhn_train_filename = "svhn_train.tfrecords"
_svhn_train_idxname = "svhn_train_idx"

_svhn_test_filename = "svhn_test.tfrecords"
_svhn_test_idxename = "svhn_test_idx"

_svhn_extra_filename = "svhn_extra.tfrecords"
_svhn_extra_idxename = "svhn_extra_idx"

_raw_root = "raw"
_processed_root = "processed"


def process_svhn(download_root, is_download=True):
    data_root = os.path.join(download_root, _data_root_name)

    raw_folder = os.path.join(data_root, _raw_root)
    processed_folder = os.path.join(data_root, _processed_root)

    if is_download:
        _download(raw_folder=raw_folder, processed_folder=processed_folder)

    if not _check_exists(processed_folder):
        raise RuntimeError('Dataset not found.' +
                           ' You can use download=True to download it')


def _download(raw_folder, processed_folder):
    if _check_exists(processed_folder):
        return

    _makedir_exist_ok(raw_folder)
    _makedir_exist_ok(processed_folder)

    download_and_extract_archive(_url, download_root=raw_folder, extract_root=processed_folder,
                                 filename=_filename, md5=_MD5)


def _makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def _check_exists(processed_folder):
    return (os.path.exists(os.path.join(processed_folder, _svhn_train_filename)) and
            os.path.exists(os.path.join(processed_folder, _svhn_train_idxname)) and
            os.path.exists(os.path.join(processed_folder, _svhn_test_filename)) and
            os.path.exists(os.path.join(processed_folder, _svhn_test_idxename)) and
            os.path.exists(os.path.join(processed_folder, _svhn_extra_filename)) and
            os.path.exists(os.path.join(processed_folder, _svhn_extra_idxename)))


if __name__ == '__main__':
    process_svhn("./datasets")
