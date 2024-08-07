import sys, os
import pytest
import numpy as np
import torch
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

@pytest.mark.fast
def test_datapaths():
    assert 'repo' in path_dict_pecl.keys()
    assert 'ukbms_images' in path_dict_pecl.keys()
    assert 'ukbms_presence' in path_dict_pecl.keys()

@pytest.mark.fast
def test_default_settings(create_ds):
    assert create_ds.zscore_im
    assert not create_ds.shuffle_order_data
    assert create_ds.species_process == 'all'
    assert create_ds.augment_image

@pytest.mark.fast
def test_bms_properties(create_ds):
    assert create_ds.n_species == 62
    assert create_ds.n_bands == 4
    assert len(create_ds.df_presence) == 1329
    assert len(create_ds.suffix_images) == 2
    
@pytest.mark.fast
def test_generic_properties(create_ds):
    assert create_ds.n_species == len(create_ds.species_list)
    assert len(create_ds.norm_means) == len(create_ds.norm_std)
    assert len(create_ds.norm_means) == create_ds.n_bands
    assert len(create_ds) == len(create_ds.df_presence)

@pytest.mark.fast
def test_load_random_data_point(create_ds):
    random_ind = np.random.randint(len(create_ds))
    print(f'Random index: {random_ind}')
    tmp = create_ds[random_ind]
    assert len(tmp) == 2
    assert type(tmp[0]) == torch.Tensor and type(tmp[1]) == torch.Tensor
    assert tmp[0].shape == (create_ds.n_bands, 224, 224)
    assert tmp[1].shape == (create_ds.n_species,)

@pytest.mark.fast
def test_find_location(create_ds):
    name_existing_loc = 'UKBMS_loc-0023'
    name_non_existing_loc = 'UKBMS_loc-9999'
    assert create_ds.find_image_path(name_existing_loc) is not None
    assert create_ds.find_image_path(name_non_existing_loc) is None
    with pytest.raises(AssertionError):
        create_ds.load_image(name_non_existing_loc)
    im = create_ds.load_image(name_existing_loc)
    assert type(im) == np.ndarray
    assert len(im.shape) == 3
    assert im.shape[0] == create_ds.n_bands

@pytest.mark.slow
def test_data_splitter(create_ds):
    splits = create_ds.split_and_save(save_indices=False, create_test=True)
    assert len(splits) == 4
    assert len(splits['clusters']) == len(create_ds)
    assert len(splits['train_indices']) + len(splits['val_indices']) + len(splits['test_indices']) == len(create_ds)
    ## assert no duplicates:
    assert len(set(splits['train_indices']) | set(splits['val_indices']) | set(splits['test_indices'])) == len(create_ds)
    ## assert no overlap:
    assert len(set(splits['train_indices']) & set(splits['val_indices'])) == 0
    assert len(set(splits['train_indices']) & set(splits['test_indices'])) == 0
    assert len(set(splits['val_indices']) & set(splits['test_indices'])) == 0

