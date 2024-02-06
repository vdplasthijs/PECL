from json import encoder
import os, sys, copy, shutil
import numpy as np
from tqdm import tqdm
import datetime
import random
import pickle
import pandas as pd 
import geopandas as gpd
import rasterio
import rasterio.features
import rioxarray as rxr
import xarray as xr
import shapely.geometry

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics
from torchvision import transforms
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

# sys.path.append(os.path.join(path_dict_pecl['repo'], 'content/'))
# import api_keys
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()
import create_dataset_utils as cdu 

def load_tiff(tiff_file_path, datatype='np', verbose=0):
    '''Load tiff file as np or da'''
    with rasterio.open(tiff_file_path) as f:
        if verbose > 0:
            print(f.profile)
        if datatype == 'np':  # handle different file types 
            im = f.read()
            assert type(im) == np.ndarray
        elif datatype == 'da':
            im = rxr.open_rasterio(f)
            assert type(im) == xr.DataArray
        else:
            assert False, 'datatype should be np or da'
    return im 

class DataSetImagePresence(torch.utils.data.Dataset):
    """Data set for image + presence/absence data. """
    def __init__(self, image_folder, presence_csv, verbose=1):
        super(DataSetImagePresence, self).__init__()
        self.image_folder = image_folder
        self.presence_csv = presence_csv
        self.verbose = verbose
        self.normalise_image = True
        self.augment_image = False
        self.shuffle_order_data = False
        self.load_data()

    def load_data(self, cols_not_species=['tuple_coords', 'n_visits', 'name_loc'],
                  prefix_name_loc='UKBMS_'):
        '''Expected format of name_loc in presence csv: UKBMS_loc-xxxxx, 
        in image folder: prefix_UKBMS_loc-xxxxx_suffix1_suffix2.tif.'''
        
        assert os.path.exists(self.presence_csv), f"Presence csv does not exist: {self.presence_csv}"
        df_presence = pd.read_csv(self.presence_csv, index_col=0)
        locs_presence = [x.lstrip(prefix_name_loc) for x in df_presence['name_loc'].values]

        assert os.path.exists(self.image_folder), f"Image folder does not exist: {self.image_folder}"
        content_image_folder = os.listdir(self.image_folder)
        locs_images = [x.split('_')[2] for x in content_image_folder]
        suffix_images = np.unique(['_'.join(x.split('_')[3:]) for x in content_image_folder])
        assert len(suffix_images) == 1, "Multiple suffixes found in image folder."
        self.suffix_images = suffix_images[0]
        prefix_images = np.unique([x.split('_')[0] for x in content_image_folder])
        assert len(prefix_images) == 1, "Multiple prefixes found in image folder."
        self.prefix_images = prefix_images[0]
        self.prefix_name_loc = prefix_name_loc

        tmp_is_present = np.array([True if x in locs_images else False for x in locs_presence])
        if self.verbose:
            print(f'Found {np.sum(tmp_is_present)} out of {len(tmp_is_present)} images in the image folder.')

        df_presence = df_presence[df_presence['name_loc'].isin([f'{prefix_name_loc}{x}' for x in locs_images])]
        assert len(df_presence) == np.sum(tmp_is_present), "Mismatch between presence/absence data and image folder."
        if self.shuffle_order_data:
            print('Shuffling data.')
            df_presence = df_presence.sample(frac=1, replace=False)
        else:
            print('Sorting data by name_loc.')
            df_presence = df_presence.sort_values(by='name_loc')
        df_presence = df_presence.reset_index(drop=True)
        self.df_presence = df_presence

        self.species_list = [x for x in self.df_presence.columns if x not in cols_not_species]
        
    def find_image_path(self, name_loc):
        im_file_name = f'{self.prefix_images}_{name_loc}_{self.suffix_images}'
        im_file_path = os.path.join(self.image_folder, im_file_name)
        return im_file_path
    
    def load_image(self, name_loc):
        im_file_path = self.find_image_path(name_loc=name_loc)
        im = load_tiff(im_file_path, datatype='np')
        
        if self.normalise_image:
            im = np.clip(im, 0, 3000)
            im = im / 3000.0
            # im = self.zscore_image(im)
        return im

    def zscore_image(self, im):
        '''Apply preprocessing function to a single image. 
        Adapted from lca.apply_zscore_preprocess_images() but more specific/faster.
        
        What would be much faster, would be to store the data already pre-processed, but
        what function to use depends on the Network.'''
        assert False, 'means and std not implemented yet.'
        im = (im - self.means) / self.std
        return im

    def transform_data(self, im, mask):
        '''https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7'''
        # Random horizontal flipping
        if random.random() > 0.5:
            im = TF.hflip(im)

        # Random vertical flipping
        if random.random() > 0.5:
            im = TF.vflip(im)

        return im
    
    def __repr__(self):
        return f"DataSetImagePresence(image_folder={self.image_folder}, presence_csv={self.presence_csv})"

    def __len__(self):
        return len(self.df_presence)
    
    def __getitem__(self, index):
        '''
        Returns im as (bands, height, width) and presence vector (species,)
        '''
        row = self.df_presence.iloc[index]
        name_loc = row.name_loc
        
        im = self.load_image(name_loc)
        im = torch.tensor(im).float()
        if self.augment_image:
            im = self.transform_data(im)

        pres_vec = row[self.species_list]
        pres_vec = torch.tensor(pres_vec.values.astype(np.float32))
        return im, pres_vec
    
class ImageEncoder(pl.LightningModule):
    '''
    Encode image using CNN Resnet + FCN.
    Train/test using various methods: PECL, or direct prediction.

    https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
    Partly inspired by https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
    '''
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, n_species, n_bands=4, n_layers_mlp=2,
                 pretrained_resnet=True, freeze_resnet=True,
                 optimizer_name='SGD', resnet_version=18,
                 lr=1e-3, batch_size=16, 
                 verbose=1):
        super(ImageEncoder, self).__init__()
        self.n_species = n_species
        self.n_bands = n_bands
        self.verbose = verbose
        self.pretrained_resnet = pretrained_resnet
        self.freeze_resnet = freeze_resnet
        self.lr = lr
        self.batch_size = batch_size
        self.resnet_version = resnet_version
        self.n_layers_mlp = n_layers_mlp

        self.optimizer_name = optimizer_name
        if optimizer_name == 'SGD':
            self.optimizer = optim.SGD
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam
        else:
            assert False, f'Optimizer {optimizer_name} not implemented.'

        self.build_model()
       
        
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()
        # self.test_acc = torchmetrics.Accuracy()

    def build_model(self):
        self.resnet = self.resnets[self.resnet_version](pretrained=self.pretrained_resnet)
        if self.n_bands == 3:
            pass 
        elif self.n_bands == 4:  # https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py
            weight = self.resnet.conv1.weight.clone()  # copy the weights from the first layer
            print(type(weight), weight.shape)
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # change the first layer to accept 4 channels
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3] = weight
                self.resnet.conv1.weight[:, 3] = weight[:, 0]
        else:
            assert False, f'Number of bands {self.n_bands} not implemented.'

        if self.freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # if tune_fc_only:  # option to only tune the fully-connected layers
        #     for child in list(self.resnet_model.children())[:-1]:
        #         for param in child.parameters():
        #             param.requires_grad = False

        if self.n_layers_mlp == 1:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.n_species)
        elif self.n_layers_mlp == 2:
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_species)
            )
        else:
            assert False, f'Number of layers {self.n_layers_mlp} not implemented.'    
    

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return self.resnet(x)
    
    def configure_optimizers(self):
        print('WARNING: configure_optimizers() not implemented yet.?')
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def pecl_pass(self, batch, distance_metric='cosine'):
        assert self.batch_size == len(batch), f'Batch size {len(batch)} does not match model batch size {self.batch_size}.'
        im, pres_vec = batch
        # im = im.to(self.device)
        # pres_vec = pres_vec.to(self.device)

        # Forward pass
        im_enc = self.forward(im)
        
        dist_array_ims = torch.zeros_like(self.batch_size * (self.batch_size - 1) // 2)
        dist_array_pres = torch.zeros_like(dist_array_ims)
        for i in range(self.batch_size):
            for j in range(i + 1, self.batch_size):
                ind_pair = i * (self.batch_size - 1) + j
                if distance_metric == 'cosine':
                    dist_array_ims[ind_pair] = 1 - F.cosine_similarity(im_enc[i], im_enc[j], dim=0)
                    dist_array_pres[ind_pair] = 1 - F.cosine_similarity(pres_vec[i], pres_vec[j], dim=0)
                elif distance_metric == 'euclidean':
                    dist_array_ims[ind_pair] = F.pairwise_distance(im_enc[i], im_enc[j], p=2)
                    dist_array_pres[ind_pair] = F.pairwise_distance(pres_vec[i], pres_vec[j], p=2)
                else:
                    assert False, f'Distance metric {distance_metric} not implemented.'
        loss_array = torch.abs(dist_array_ims - dist_array_pres)
        loss = torch.mean(loss_array)
        return loss
                
    def training_step(self, batch, batch_idx):
        loss = self.pecl_pass(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.pecl_pass(batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.pecl_pass(batch)
        self.log('test_loss', loss)
        return loss
    