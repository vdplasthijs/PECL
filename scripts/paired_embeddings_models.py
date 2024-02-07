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
from pytorch_lightning import loggers as pl_loggers


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
    def __init__(self, image_folder, presence_csv, shuffle_order_data=False,
                 species_process='all',
                 augment_image=False, verbose=1):
        super(DataSetImagePresence, self).__init__()
        self.image_folder = image_folder
        self.presence_csv = presence_csv
        self.verbose = verbose
        self.normalise_image = True
        self.augment_image = augment_image
        self.shuffle_order_data = shuffle_order_data
        self.species_process = species_process
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

        original_species_list = [x for x in df_presence.columns if x not in cols_not_species]
        n_original_species = len(original_species_list)
        if self.species_process == 'all':
            pass 
        # elif self.species_process == 'only_present':
        #     cols_species_present = [x for x in original_species_list if np.sum(df_presence[x]) > 0]
        #     cols_keep = cols_not_species + cols_species_present
        #     df_presence = df_presence[cols_keep]
        #     print(f'Only keeping {len(cols_species_present)}/{len(original_species_list)} species with at least one record present.')
        elif self.species_process == 'priority_species':
            priority_species = ['Carterocephalus palaemon', 'Thymelicus acteon', 'Leptidea sinapis',  # 'Leptidea juvernica', 
                                'Coenonympha tullia',
                                # 'Boloria euphrosyne', 
                                'Fabriciana adippe', 'Euphydryas aurinia',
                                # 'Melitaea athalia', 
                                'Hamearis lucina',
                                # 'Phengaris arion',
                                  'Aricia artaxerxes']  ## From BC 2022 report: These UK Priority Species of butterflies are Chequered Skipper, Lulworth Skipper, Wood White, Cryptic Wood White, Large Heath, Pearl-bordered Fritillary, High Brown Fritillary, Marsh Fritillary, Heath Fritillary, Duke of Burgundy, Large Blue and Northern Brown Argus
            for sp in priority_species:
                assert sp in original_species_list, f'Indicator species {sp} not found in species list.'
            cols_keep = cols_not_species + priority_species
            df_presence = df_presence[cols_keep]
            print(f'Only keeping {len(priority_species)}/{len(original_species_list)} species that are indicator species.')
            # assert False, 'Not implemented yet.'
        elif self.species_process == 'top_20':
            obs_per_species = df_presence[original_species_list].sum(axis=0)
            inds_sort = np.argsort(obs_per_species)
            cols_species_top20 = inds_sort[-20:]
            cols_keep = cols_not_species + [original_species_list[x] for x in cols_species_top20]
            df_presence = df_presence[cols_keep]
            print(f'Only keeping top 20 species with most observations.')
        elif self.species_process == 'pca':
            n_pcs_keep = 16
            ## get PCA of species data
            assert False, 'Not implemented yet.'
        else:
            assert False, f'Species process {self.species_process} not implemented.'

        self.species_list = [x for x in df_presence.columns if x not in cols_not_species]
        self.df_presence = df_presence
        self.n_species = len(self.species_list)
    
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

    def __init__(self, n_species, n_enc_channels=32, n_bands=4, n_layers_mlp=2,
                 pretrained_resnet=True, freeze_resnet=True,
                 optimizer_name='SGD', resnet_version=18,
                 pecl_distance_metric='cosine',
                 lr=1e-3, #  batch_size=16, 
                 training_method='pecl',
                 verbose=1):
        super(ImageEncoder, self).__init__()
        self.n_species = n_species
        self.n_enc_channels = n_enc_channels
        self.n_bands = n_bands
        self.verbose = verbose
        self.pretrained_resnet = pretrained_resnet
        self.freeze_resnet = freeze_resnet
        self.lr = lr
        # self.batch_size = batch_size
        self.resnet_version = resnet_version
        self.n_layers_mlp = n_layers_mlp
        self.pecl_distance_metric = pecl_distance_metric

        self.optimizer_name = optimizer_name
        if optimizer_name == 'SGD':
            self.optimizer = optim.SGD
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam
        else:
            assert False, f'Optimizer {optimizer_name} not implemented.'

        self.build_model()

        if training_method == 'pecl':
            self.forward_pass = self.pecl_pass
        elif training_method == 'pred':
            self.forward_pass = self.pred_pass
        elif training_method == 'pred_incl_enc':
            self.forward_pass = self.pred_pass_incl_encoding_model
        else:
            assert False, f'Training method {training_method} not implemented.'
       
    def build_model(self):
        ## Load Resnet, if needed modify first layer to accept 4 bands
        self.resnet = self.resnets[self.resnet_version](pretrained=self.pretrained_resnet)
        if self.n_bands == 3:
            pass 
        elif self.n_bands == 4:  # https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py
            weight = self.resnet.conv1.weight.clone()  # copy the weights from the first layer
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # change the first layer to accept 4 channels
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3] = weight
                self.resnet.conv1.weight[:, 3] = weight[:, 0]
        else:
            assert False, f'Number of bands {self.n_bands} not implemented.'

        ## Modify last layer to output n_enc_channels
        if self.n_layers_mlp == 1:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.n_enc_channels)
        elif self.n_layers_mlp == 2:
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_enc_channels)
            )
        else:
            assert False, f'Number of layers {self.n_layers_mlp} not implemented.'    

        ## Freeze Resnet, except for self.resnet.fc, if requested:
        if self.freeze_resnet:
            for child in list(self.resnet.children())[:-1]:  # Freeze all layers except the last one
                for param in child.parameters():  # set all parameters to not require gradients
                    param.requires_grad = False

        ## Prediction model to predict presence/absence from encoded image
        self.prediction_model = nn.Sequential(
            nn.Linear(self.n_enc_channels, self.n_species),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return self.resnet(x)
    
    def configure_optimizers(self):
        print('WARNING: configure_optimizers() not implemented yet.?')
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def pecl_pass(self, batch):
        '''Train the encoding model using the PECL method.'''
        im, pres_vec = batch
        # print(im.shape, pres_vec.shape, self.batch_size, len(im), len(pres_vec))
        curr_batch_size = len(im)  # would normally be self.batch_size, but last batch might be smaller
        # im = im.to(self.device)
        # pres_vec = pres_vec.to(self.device)

        # Forward pass
        im_enc = self.forward(im)
        
        ## Calculate distance between pairs of images and pairs of presence vectors
        dist_array_ims = torch.zeros(curr_batch_size * (curr_batch_size - 1) // 2)
        dist_array_pres = torch.zeros_like(dist_array_ims)
        ind_pair = -1
        for i in range(curr_batch_size):
            for j in range(i + 1, curr_batch_size):  # avoid duplicates
                ind_pair += 1
                # print(ind_pair, i, j, im[i].shape, im[j].shape, im_enc[i].shape, im_enc[j].shape)
                if self.pecl_distance_metric == 'cosine':
                    dist_array_ims[ind_pair] = 1 - F.cosine_similarity(im_enc[i], im_enc[j], dim=0)
                    dist_array_pres[ind_pair] = 1 - F.cosine_similarity(pres_vec[i], pres_vec[j], dim=0)
                elif self.pecl_distance_metric == 'euclidean':
                    dist_array_ims[ind_pair] = F.pairwise_distance(im_enc[i], im_enc[j], p=2)
                    dist_array_pres[ind_pair] = F.pairwise_distance(pres_vec[i], pres_vec[j], p=2)
                else:
                    assert False, f'Distance metric {self.pecl_distance_metric} not implemented.'
        loss_array = torch.abs(dist_array_ims - dist_array_pres)  # L1 loss
        loss = torch.mean(loss_array)
        return loss
    
    def pred_pass(self, batch):
        '''Only trains the prediction model, not the encoding model.'''
        im, pres_vec = batch
        # im = im.to(self.device)
        # pres_vec = pres_vec.to(self.device)

        # Forward pass
        with torch.no_grad():  # Don't train encoding model here. 
            im_enc = self.forward(im)
        pres_pred = self.prediction_model(im_enc)
        loss = F.mse_loss(pres_pred, pres_vec) 
        return loss
    
    def pred_pass_incl_encoding_model(self, batch):
        '''Trains the prediction model and encoding model (but using regular SL back-prop, not PECL).'''
        im, pres_vec = batch
        # im = im.to(self.device)
        # pres_vec = pres_vec.to(self.device)

        # Forward pass
        im_enc = self.forward(im)
        pres_pred = self.prediction_model(im_enc)
        loss = F.mse_loss(pres_pred, pres_vec) 
        return loss
                
    def training_step(self, batch, batch_idx):
        loss = self.forward_pass(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward_pass(batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.forward_pass(batch)
        self.log('test_loss', loss)
        return loss
    

def train_pecl(n_enc_channels=32, n_layers_mlp=2, pretrained_resnet=True, freeze_resnet=True,
               resnet_version=18, pecl_distance_metric='cosine',
               training_method='pecl', lr=1e-3, batch_size=8, n_epochs_max=10, 
               image_folder=None, presence_csv=None,
               verbose=1, fix_seed=42, use_mps=True):
    if fix_seed is not None:
        pl.seed_everything(fix_seed)

    if image_folder is None:
        image_folder = '/Users/t.vanderplas/data/UKBMS_sent2_ds/sent2-4band/2019/m-06-09/'
    if presence_csv is None:
        presence_csv = '/Users/t.vanderplas/data/UKBMS_sent2_ds/bms_presence/bms_presence_y-2018-2019_th-200.csv'

    if use_mps:
        assert torch.backends.mps.is_available()
        assert torch.backends.mps.is_built()
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/Users/t.vanderplas/models/PECL')
        n_cpus = 8
        # acc_use = 'mps'
        acc_use = 'cpu'
        folder_save = '/Users/t.vanderplas/models/PECL/'
    else:
        assert torch.cuda.is_available(), 'No GPU available.'
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='/home/tplas/models/')
        n_cpus = 8
        acc_use = 'gpu'
        folder_save = '/home/tplas/models/PECL/'
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
        print(f'Created folder {folder_save}.')
    if verbose > 0:  # possibly also insert assert versions
        print(f'Pytorch version is {torch.__version__}') 

    ds = DataSetImagePresence(image_folder=image_folder, presence_csv=presence_csv,
                              shuffle_order_data=True)
    train_ds, val_ds = torch.utils.data.random_split(ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))])
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus, persistent_workers=True) #drop_last=True, pin_memory=True
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=n_cpus, persistent_workers=True) #drop_last=True, pin_memory=True

    model = ImageEncoder(n_species=ds.n_species, n_enc_channels=n_enc_channels, n_bands=4, n_layers_mlp=n_layers_mlp,
                        pretrained_resnet=pretrained_resnet, freeze_resnet=freeze_resnet,
                        optimizer_name='SGD', resnet_version=resnet_version,
                        pecl_distance_metric=pecl_distance_metric,
                        lr=lr,  # batch_size=batch_size, 
                        training_method=training_method,
                        verbose=verbose)
    
    # cb_metrics = cl.MetricsCallback()
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                            filename="best_checkpoint_val-{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}")]

    trainer = pl.Trainer(max_epochs=n_epochs_max, accelerator=acc_use, progress_bar_refresh_rate=20,
                         callbacks=callbacks, logger=tb_logger)

    timestamp_start = datetime.datetime.now()
    print(f'-- Starting training at {timestamp_start} with {n_epochs_max} epochs.')
    trainer.fit(model, train_dl, val_dl)

    timestamp_end = datetime.datetime.now()
    print(f'-- Finished training at {timestamp_end}.')
    return model