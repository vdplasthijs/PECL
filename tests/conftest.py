import sys, os
import pytest
# import numpy as np
# import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from DataSetImagePresence import DataSetImagePresence
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()

@pytest.fixture(params=['val', 'train'])
def create_ds(request):
    mode = request.param
    ds = DataSetImagePresence(image_folder=path_dict_pecl['ukbms_images'],
                              presence_csv=path_dict_pecl['ukbms_presence'],
                              mode=mode)
    return ds


@pytest.fixture
def get_split_path():
    filepath_train_val_split = os.path.join(path_dict_pecl['repo'], 
                                            'content/split_indices_2024-03-04-1831.pth')
    return filepath_train_val_split

@pytest.fixture
def create_split_ds(create_ds, get_split_path):
    train_ds, val_ds, test_ds = create_ds.split_into_train_val(filepath=get_split_path)
    return (train_ds, val_ds, test_ds)

