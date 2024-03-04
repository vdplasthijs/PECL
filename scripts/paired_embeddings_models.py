from json import encoder
import os, sys, copy, shutil, ast
import numpy as np
from tqdm import tqdm
import datetime, pickle
import random
import pickle
import pandas as pd 
import geopandas as gpd
import rasterio
import rasterio.features
import rioxarray as rxr
import xarray as xr
import shapely.geometry
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA 
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit
from geopy.distance import distance as geodist # avoid naming confusion
from scipy.spatial.distance import cdist

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics
from torchvision.transforms import v2
# from torchvision import transforms
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


# sys.path.append(os.path.join(path_dict_pecl['repo'], 'content/'))
# import api_keys
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()
import create_dataset_utils as cdu 
from load_seco_resnet import map_seco_to_torchvision_weights

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
                 species_process='all', n_bands=4, zscore_im=True,
                 mode='train',
                 augment_image=True, verbose=1):
        super(DataSetImagePresence, self).__init__()
        self.image_folder = image_folder
        self.presence_csv = presence_csv
        self.mode = mode
        self.verbose = verbose
        self.zscore_im = zscore_im
        if self.zscore_im:
            ## Values obtained from full data set (1336 images):
            self.norm_means = np.array([661.1047,  770.6800,  531.8330, 3228.5588]).astype(np.float32) 
            self.norm_std = np.array([640.2482,  571.8545,  597.3570, 1200.7518]).astype(np.float32) 
            self.norm_means = self.norm_means[:, None, None]
            self.norm_std = self.norm_std[:, None, None]
        else:
            self.norm_means = None
            self.norm_std = None
        self.augment_image = augment_image
        self.shuffle_order_data = shuffle_order_data
        self.species_process = species_process
        self.n_bands = n_bands
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
        self.suffix_images = suffix_images
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
        elif self.species_process == 'priority_species' or self.species_process == 'priority_species_present':
            # priority_species = ['Carterocephalus palaemon', 'Thymelicus acteon', 'Leptidea sinapis',  # 'Leptidea juvernica', 
            #                     'Coenonympha tullia',
            #                     # 'Boloria euphrosyne', 
            #                     'Fabriciana adippe', 'Euphydryas aurinia',
            #                     # 'Melitaea athalia', 
            #                     'Hamearis lucina',
            #                     # 'Phengaris arion',
            #                       'Aricia artaxerxes']  ## From BC 2022 report: These UK Priority Species of butterflies are Chequered Skipper, Lulworth Skipper, Wood White, Cryptic Wood White, Large Heath, Pearl-bordered Fritillary, High Brown Fritillary, Marsh Fritillary, Heath Fritillary, Duke of Burgundy, Large Blue and Northern Brown Argus
            priority_species = ['Pararge aegeria', 'Maniola jurtina', 'Coenonympha pamphilus']
            
            for sp in priority_species:
                assert sp in original_species_list, f'Indicator species {sp} not found in species list.'
            cols_keep = cols_not_species + priority_species
            df_presence = df_presence[cols_keep]
            print(f'Only keeping {len(priority_species)}/{len(original_species_list)} species that are indicator species.')
            if self.species_process == 'priority_species_present':
                ## change values to 1 if present, 0 if not
                df_presence[priority_species] = df_presence[priority_species].applymap(lambda x: 1 if x > 0 else 0)
                print(f'Changing presence/absence to 1/0 for priority species.')
            n_locs_at_least_one_present = np.sum(df_presence[priority_species].sum(axis=1) > 0)
            print(f'At least one priority species present in {n_locs_at_least_one_present} out of {len(df_presence)} locations.')
        elif self.species_process == 'top_20':
            obs_per_species = df_presence[original_species_list].sum(axis=0)
            inds_sort = np.argsort(obs_per_species)
            cols_species_top20 = inds_sort[-20:]
            cols_keep = cols_not_species + [original_species_list[x] for x in cols_species_top20]
            df_presence = df_presence[cols_keep]
            print(f'Only keeping top 20 species with most observations.')
        elif self.species_process == 'pca':
            n_pcs_keep = 20
            pca = PCA(n_components=n_pcs_keep)
            pca.fit(df_presence[original_species_list].values)
            df_presence_pca = pd.DataFrame(pca.transform(df_presence[original_species_list].values))
            df_presence_pca.columns = [f'PCA_{x}' for x in range(n_pcs_keep)]

            ## normalise to 0-1 range
            min_val = df_presence_pca.min().min()
            max_val = df_presence_pca.max().max()
            df_presence_pca = (df_presence_pca - min_val) / (max_val - min_val)
            self.pca_min_val = min_val
            self.pca_max_val = max_val
            self.pca_components = pca.components_
            self.pca = pca

            df_presence = pd.concat([df_presence[cols_not_species], df_presence_pca], axis=1)
            total_expl_var = np.sum(pca.explained_variance_ratio_)
            print(f'PCA with {n_pcs_keep} components explains {100 * total_expl_var:.1f}% of the variance.')
        else:
            assert False, f'Species process {self.species_process} not implemented.'

        self.species_list = [x for x in df_presence.columns if x not in cols_not_species]
        self.df_presence = df_presence
        self.n_species = len(self.species_list)

        ## determine weights:
        total = self.df_presence[self.species_list].sum().sum()
        self.weights = 1 / (self.df_presence[self.species_list].sum(0) / total)
        ## clip:
        self.weights = np.clip(self.weights, np.percentile(self.weights, 5), np.percentile(self.weights, 75))
        self.weights = self.weights / np.min(self.weights)
        assert np.all(self.weights.index == self.species_list), f'Index of weights {self.weights.index} does not match species list {self.species_list}.'
        # self.weights_values = torch.tensor(self.weights.values).float()
        self.weights_values = self.weights.values

    def find_image_path(self, name_loc):
        
        if len(self.suffix_images) == 1:
            im_file_name = f'{self.prefix_images}_{name_loc}_{self.suffix_images[0]}'
            im_file_path = os.path.join(self.image_folder, im_file_name)
            return im_file_path
        else:
            for s in self.suffix_images:
                im_file_name = f'{self.prefix_images}_{name_loc}_{s}'
                im_file_path = os.path.join(self.image_folder, im_file_name)
                if os.path.exists(im_file_path):
                    return im_file_path
    
    def load_image(self, name_loc):
        im_file_path = self.find_image_path(name_loc=name_loc)
        im = load_tiff(im_file_path, datatype='np')
        
        if self.n_bands == 4:
            pass 
        elif self.n_bands == 3:
            im = im[:3, :, :]
        else:
            assert False, f'Number of bands {self.n_bands} not implemented.'

        if self.zscore_im:
            im = im.astype(np.int32)
            im = self.zscore_image(im)
        else:
            # if self.n_bands == 4:
            #     print('WARNING: Clipping image to 0-3000 range, but NIR band average EXCEEDS max value typically.')
            im = np.clip(im, 0, 3000)
            im = im / 3000.0
        return im

    def zscore_image(self, im):
        '''Apply preprocessing function to a single image. 
        raw_sent2_means = torch.tensor([661.1047,  770.6800,  531.8330, 3228.5588])
        raw_sent2_stds = torch.tensor([640.2482,  571.8545,  597.3570, 1200.7518])
        '''
        im = (im - self.norm_means) / self.norm_std
        return im

    def transform_data(self, im):
        '''Apply random augmentations to the image.
        From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
        '''
        if self.mode == 'train':
            augment_transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                                         v2.RandomVerticalFlip(p=0.5),  
                                        #  v2.RandomResizedCrop(size=224),
                                        v2.RandomCrop(size=224),
                                         ])
            
            # augment_transforms_3band = v2.Compose([
            #                               v2.RandomApply([
            #                                   v2.ColorJitter(brightness=0.5,
            #                                                 contrast=0.5,
            #                                                 saturation=0.5,
            #                                                 hue=0.1)
            #                                                 ], p=0.8),
            #                             #   v2.RandomGrayscale(p=0.2),
            #                               v2.GaussianBlur(kernel_size=9),
            #                              ])
            im = augment_transforms(im)
            # im[:3, :, :] = augment_transforms_3band(im[:3, :, :])
        elif self.mode == 'val':
            assert im.shape[1] == im.shape[2] == 256, f'Image shape {im.shape} not 256x256.'
            im = im[:, 16:240, 16:240]  # Crop to centre 224x224  #TODO; use v2 CentreCrop
        
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
    
    def plot_image(self, index=None, loc_name=None, ax=None, 
                   plot_species_bar=True):
        if loc_name is not None and index is None:
            if loc_name in self.df_presence['name_loc'].values:
                index = self.df_presence[self.df_presence['name_loc'] == loc_name].index[0]
            else:
                # assert False, f'Location {loc_name} not found.'
                print( f'Location {loc_name} not found in data set.')
                return None
        elif index is not None and loc_name is None:
            loc_name = self.df_presence.iloc[index]['name_loc']
        else: 
            assert False, 'Either index or loc_name must be provided.'

        self.zscore_im = False
        im, pres_vec = self.__getitem__(index)
        self.zscore_im = True
        if len(im) == 4:
            im = im[0:3]
        elif len(im) == 3:
            pass
        else:
            assert False, f'Number of bands {len(im)} not implemented.'
        im = im.numpy()

        if ax is None:
            ax = plt.subplot(111)
        if type(im) == xr.DataArray:
            plot_im = im.to_numpy()
        else:
            plot_im = im
        use_im_extent = False
        if use_im_extent:
            extent = [im.x.min(), im.x.max(), im.y.min(), im.y.max()]
        else:
            extent = None
        rasterio.plot.show(plot_im, ax=ax, cmap='viridis', 
                        extent=extent, vmin=0, vmax=1)
        for sp in ax.spines:
            ax.spines[sp].set_visible(False)
        ax.set_aspect('equal')
        ax.set_title(f'{loc_name}, id {index}')

        if plot_species_bar:
            species_vector = self.df_presence[self.df_presence.name_loc == loc_name][self.species_list]
            species_vector = species_vector.values[0]
            assert len(species_vector) == len(self.species_list), f'Length of species vector {len(species_vector)} does not match number of species {len(self.species_list)}.'

            divider = make_axes_locatable(ax)
            targ_ax = divider.append_axes('right', size='5%', pad=0.01)
            targ_ax.imshow(species_vector[:, None], cmap='Greys', aspect='auto', 
                           interpolation='nearest', vmin=0, vmax=1)
            targ_ax.set_xticks([])
            targ_ax.set_yticks([])
        else:
            targ_ax = None

        return ax, targ_ax

    def determine_mean_std_entire_ds(self, max_iter=100):
        for i_sample, sample in tqdm(enumerate(self)):
            im, target = sample
            if i_sample == 0:
                im_aggr = im[None, ...].clone()
            else:
                im_aggr = torch.cat((im_aggr, im[None, ...]), dim=0)
            if i_sample == max_iter:
                print(f'Breaking after {max_iter} samples.')
                break 

        im_aggr.shape
        mean = im_aggr.mean(dim=(0, 2, 3))
        std = im_aggr.std(dim=(0, 2, 3))
        print(mean, std)
        return mean, std

    def split_into_train_val(self, filepath=None):
        assert filepath is not None 
        assert os.path.exists(filepath), f'Filepath {filepath} does not exist.'
        split_indices = torch.load(filepath)
        train_ds = torch.utils.data.Subset(self, split_indices['train_indices'])
        val_ds = torch.utils.data.Subset(self, split_indices['val_indices'])

        train_ds.dataset.mode = 'train'
        val_ds.dataset.mode = 'val'
        if 'test_indices' in split_indices.keys():
            test_ds = torch.utils.data.Subset(self, split_indices['test_indices'])
            test_ds.dataset.mode = 'val'
        else:
            test_ds = None
        return train_ds, val_ds, test_ds
    
    def split_and_save(self, split_fun='spatial_clusters', create_test=True):
        if split_fun == 'random':
            print('WARNING: splitting randomly')
            assert False, 'This is not implemented.'
            train_ds, val_ds = torch.utils.data.random_split(self, [int(0.8 * len(self)), len(self) - int(0.8 * len(self))])
            # Save the split indices
            split_indices = {
                'train_indices': train_ds.indices,
                'val_indices': val_ds.indices
            }
        elif split_fun == 'spatial_clusters':
            print('Splitting based on spatial clusters. This can take a little while.')
            coords = self.df_presence.tuple_coords
            coords = [ast.literal_eval(loc) for loc in coords]
            coords = np.array(coords)
            # coords_points = [shapely.geometry.Point(loc) for loc in coords]
            if len(self.df_presence) > 2000:
                assert False, 'You can ignore this, but be aware that this might be slow.'

            ## 4000 m distance between points. Use geodist to calculate true distance.
            clustering = DBSCAN(eps=4000, metric=lambda u, v: geodist(u, v).meters, min_samples=2).fit(coords)

            ## Non-clustered points are labeled -1. Change to new cluster label.
            clusters = copy.deepcopy(clustering.labels_)
            new_cl = np.max(clusters) + 1
            for i, cl in enumerate(clusters):
                if cl == -1:
                    clusters[i] = new_cl
                    new_cl += 1

            if create_test:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
                train_val_inds, test_inds = next(gss.split(np.arange(len(coords)), groups=clusters))
                gss_2 = GroupShuffleSplit(n_splits=1, test_size=(0.15 / 0.85), random_state=0)
                tmp_train_inds, tmp_val_inds = next(gss_2.split(train_val_inds, groups=clusters[train_val_inds]))
                train_inds = train_val_inds[tmp_train_inds]
                val_inds = train_val_inds[tmp_val_inds]
                clusters_train = clusters[train_inds]
                clusters_val = clusters[val_inds]
                clusters_test = clusters[test_inds]
                ## assert no overlap in indices:
                assert len(np.intersect1d(train_inds, val_inds)) == 0, np.intersect1d(train_inds, val_inds)
                assert len(np.intersect1d(train_inds, test_inds)) == 0, np.intersect1d(train_inds, test_inds)
                assert len(np.intersect1d(val_inds, test_inds)) == 0, np.intersect1d(val_inds, test_inds)

                ## assert no overlap in clusters:
                assert len(np.intersect1d(clusters_train, clusters_val)) == 0, np.intersect1d(clusters_train, clusters_val)
                assert len(np.intersect1d(clusters_train, clusters_test)) == 0, np.intersect1d(clusters_train, clusters_test)
                assert len(np.intersect1d(clusters_val, clusters_test)) == 0, np.intersect1d(clusters_val, clusters_test)

                print(f'Create {len(train_inds)} train, {len(val_inds)} val, {len(test_inds)} test indices.')
                split_indices = {'train_indices': train_inds,
                                'val_indices': val_inds,
                                'test_indices': test_inds,
                                'clusters': clusters}
            else:
                ## Split into train and test:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                train_inds, test_inds = next(gss.split(np.arange(len(coords)), groups=clusters))

                clusters_train = clusters[train_inds]
                clusters_test = clusters[test_inds]
                assert len(np.intersect1d(train_inds, test_inds)) == 0, np.intersect1d(train_inds, test_inds)
                assert len(np.intersect1d(clusters_train, clusters_test)) == 0, np.intersect1d(clusters_train, clusters_test)

                split_indices = {'train_indices': train_inds,
                                  'val_indices': test_inds,
                                  'clusters': clusters}
                print(f'Create {len(train_inds)} train, {len(test_inds)} val indices.')
        timestamp = cdu.create_timestamp()
        torch.save(split_indices, os.path.join(path_dict_pecl['repo'], f'content/split_indices_{timestamp}.pth'))
        print(f'Saved split indices to split_indices_{timestamp}.pth')
        return split_indices

class ImageEncoder(pl.LightningModule):
    '''
    Encode image using CNN Resnet + FCN.
    Train/test using various methods: PECL, or direct prediction.

    https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
    Partly inspired by https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
    '''
    resnets = {
        18: models.resnet18,
        # 34: models.resnet34,
        50: models.resnet50,
        # 101: models.resnet101,
        # 152: models.resnet152,
    }

    def __init__(self, n_species=62, n_enc_channels=128, n_bands=4, 
                 n_layers_mlp_resnet=1, n_layers_mlp_pred=2,
                 pretrained_resnet='imagenet', freeze_resnet=True,
                 optimizer_name='Adam', resnet_version=18,
                 pecl_distance_metric='softmax',
                 pred_train_loss='mse', class_weights=None,
                 lr=1e-3, pecl_knn=5, pecl_knn_hard_labels=False,
                 training_method='pecl', alpha_ratio_loss=None,
                 normalise_embedding='l2', use_mps=True,
                 use_lr_scheduler=False, seed_used=None, 
                 batch_size_used=None, p_dropout=0,
                 verbose=1, time_created=None):
        super(ImageEncoder, self).__init__()
        self.save_hyperparameters()
        self.n_species = n_species
        self.n_enc_channels = n_enc_channels
        self.n_bands = n_bands
        assert self.n_bands in [3, 4], f'Number of bands {self.n_bands} not implemented.'
        self.verbose = verbose
        self.seed_used = seed_used  # Save seed used in training function
        self.batch_size_used = batch_size_used  ## only saved for reference, not required
        self.pretrained_resnet = pretrained_resnet
        self.freeze_resnet = freeze_resnet
        self.lr = lr
        self.alpha_ratio_loss = alpha_ratio_loss
        self.resnet_version = resnet_version
        self.n_layers_mlp_resnet = n_layers_mlp_resnet
        assert self.n_layers_mlp_resnet == 1, 'Expecting 1 layer MLP for projection head for now.'
        # assert self.n_layers_mlp_resnet in [1, 2], f'Number of MLP layers {self.n_layers_mlp_resnet} not implemented.'
        self.n_layers_mlp_pred = n_layers_mlp_pred
        assert self.n_layers_mlp_pred in [1, 2, 3], 'Expecting 1 or 2 or 3 layer MLP for prediction head for now.'
        self.pecl_distance_metric = pecl_distance_metric
        self.optimizer_name = optimizer_name
        self.use_mps = use_mps
        self.normalise_embedding = normalise_embedding
        assert self.normalise_embedding == 'l2', 'Currently expecting l2 normalisation'
        self.pecl_knn = pecl_knn
        self.pecl_knn_hard_labels = pecl_knn_hard_labels
        self.use_lr_scheduler = use_lr_scheduler
        self.use_dropout = p_dropout > 0
        self.p_dropout = p_dropout
        if self.use_dropout:
            print('Using dropout.')
        self.model_name = None
        self.v_num = None
        self.log_dir = None
        self.time_created = time_created
        self.description = f'ImageEncoder with {n_enc_channels} encoding channels, {n_bands} bands, {n_species} species, {n_layers_mlp_resnet} MLP layers, {resnet_version} Resnet, {pecl_distance_metric} distance metric, {training_method} training method.'
        self.df_metrics = None 
        self.build_class_weights(class_weights=class_weights)
        self.build_model()
        self.build_training_method(training_method=training_method, pred_train_loss=pred_train_loss)
       
        if self.pred_train_loss in ['weighted-bce', 'weighted-ce']:
            assert self.class_weights is not None, 'Class weights not set.'

    def __str__(self) -> str:
        return super().__str__() + f' {self.description}'
    
    def __repr__(self) -> str:
        return super().__repr__() + f' {self.description}'

    def change_description(self, new_description='', add=True):
        '''Just used for keeping notes etc.'''
        if add:
            self.description = self.description + '\n' + new_description
        else:
            self.description = new_description

    def build_class_weights(self, class_weights):
        ## Load class weights
        if class_weights is not None:
            assert class_weights.ndim == 1, f'Class weights shape {class_weights.shape} not 1D.'
            assert class_weights.shape[0] == self.n_species, f'Class weights shape {class_weights.shape} does not match number of species {self.n_species}.'
            self.class_weights = torch.tensor(class_weights).float() 
            print(f'Loaded {self.class_weights.shape[0]} class weights on {self.class_weights.device}.')
            if self.use_mps:
                self.class_weights = self.class_weights.to('mps')
                print(f'Class weights now on {self.class_weights.device}.')
        else:
            print('No class weights.')
            self.class_weights = None

        ## Some checks:
        if self.class_weights is not None:
            assert self.class_weights is not None, 'Class weights not set.'
            assert type(self.class_weights) == torch.Tensor, f'Class weights type {type(self.class_weights)} not torch.Tensor.'
            assert self.class_weights.ndim == 1, f'Class weights shape {self.class_weights.shape} not 1D.'
            assert self.class_weights.shape[0] == self.n_species, f'Class weights shape {self.class_weights.shape} does not match number of species {self.n_species}.'
        
    def build_training_method(self, training_method, pred_train_loss):
        self.train_im_enc_during_pred = False
        if training_method == 'pecl':
            self.forward_pass = self.pecl_pass
            self.pred_train_loss = None
            self.name_train_loss = f'pecl-{self.pecl_distance_metric}'
            self.freeze_resnet_layers(freeze_all_but_last=self.freeze_resnet,
                                      freeze_last=False)
            self.freeze_prediction_model(freeze=True)
        elif training_method == 'pred':
            self.forward_pass = self.pred_pass
            self.pred_train_loss = pred_train_loss
            self.name_train_loss = f'pred-{pred_train_loss}'
            self.freeze_resnet_layers(freeze_all_but_last=True,
                                      freeze_last=True)
            self.freeze_prediction_model(freeze=False)
        elif training_method == 'pred_incl_enc':
            self.forward_pass = self.pred_pass
            self.pred_train_loss = pred_train_loss
            self.name_train_loss = f'pred-{pred_train_loss}'
            self.train_im_enc_during_pred = True
            self.freeze_resnet_layers(freeze_all_but_last=self.freeze_resnet,
                                      freeze_last=False)
            self.freeze_prediction_model(freeze=False)
        elif training_method == 'pred_and_pecl':
            self.forward_pass = self.pred_and_pecl_pass
            self.pred_train_loss = pred_train_loss
            self.name_train_loss = f'pred-{pred_train_loss}_pecl-{self.pecl_distance_metric}'
            assert self.alpha_ratio_loss is not None, 'Expecting alpha_ratio_loss to be set.'
            assert self.alpha_ratio_loss >= 0, 'Expecting alpha_ratio_loss to be >= 0.'    
            
            self.freeze_resnet_layers(freeze_all_but_last=self.freeze_resnet,
                                      freeze_last=False)
            self.freeze_prediction_model(freeze=False)    
        else:
            assert False, f'Training method {training_method} not implemented.'

    def build_model(self):
        ## Load Resnet, if needed modify first layer to accept 4 bands
        if self.pretrained_resnet == 'seco':
            self.resnet = map_seco_to_torchvision_weights(model=None, device_use='mps' if self.use_mps else 'cpu',
                                                          resnet_name=f'resnet{self.resnet_version}', verbose=0)
            self.pretrained_weights_name = f'seco_resnet{self.resnet_version}_1m'
            print('Loaded Resnet with SeCo weights.')
        elif self.pretrained_resnet is None or self.pretrained_resnet == False:
            self.resnet = self.resnets[self.resnet_version](weights=None)
            print(f'Loaded Resnet{self.resnet_version} with random weights.')
        elif self.pretrained_resnet == 'imagenet' or self.pretrained_resnet == True:
            self.pretrained_weights_name = "IMAGENET1K_V1"
            self.resnet = self.resnets[self.resnet_version](weights=self.pretrained_weights_name)
            print(f'Loaded Resnet{self.resnet_version} with {self.pretrained_weights_name} weights.')
        else:
            raise ValueError(f'Pretrained resnet {self.pretrained_resnet} not implemented.')
        
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
        if self.n_layers_mlp_resnet == 1:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.n_enc_channels)
        elif self.n_layers_mlp_resnet == 2:
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_enc_channels)
            )
        else:
            assert False, f'Number of layers {self.n_layers_mlp_resnet} not implemented.'    

        ## Prediction model to predict presence/absence from encoded image
        if self.n_layers_mlp_pred == 1:
            self.prediction_model = nn.Sequential(
                nn.Linear(self.n_enc_channels, self.n_species),
                nn.Sigmoid())
        elif self.n_layers_mlp_pred == 2:
            self.prediction_model = nn.Sequential(
                nn.Linear(self.n_enc_channels, self.n_enc_channels),
                nn.ReLU(),
                nn.Linear(self.n_enc_channels, self.n_species),
                nn.Sigmoid())
        elif self.n_layers_mlp_pred == 3:
            self.prediction_model = nn.Sequential(
                nn.Linear(self.n_enc_channels, self.n_enc_channels),
                nn.ReLU(),
                nn.Linear(self.n_enc_channels, self.n_enc_channels),
                nn.ReLU(),
                nn.Linear(self.n_enc_channels, self.n_species),
                nn.Sigmoid())
        else:
            raise ValueError(f'Number of layers {self.n_layers_mlp_pred} not implemented.')

    def freeze_resnet_layers(self, freeze_all_but_last=True, freeze_last=False):
        layers_resnet = list(self.resnet.children())
        n_layers = len(layers_resnet)
        for i_c, child in enumerate(layers_resnet):
            if i_c < n_layers - 1:  # everything except last layer , which is the FC layer
                for param in child.parameters():
                    if freeze_all_but_last:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for param in child.parameters():
                    if freeze_last:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        print(f'Freezing all but last layer: {freeze_all_but_last}, last layer: {freeze_last}.')

    def freeze_prediction_model(self, freeze=True):
        for param in self.prediction_model.parameters():
            param.requires_grad = not freeze
        print(f'Freezing prediction model: {freeze}.')

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        encoding = self.resnet(x)
        if self.normalise_embedding is None:
            assert False, 'Expecting normalisation of embedding.'
        elif self.normalise_embedding == 'l2':
            encoding = F.normalize(encoding, p=2, dim=1)  ## dim=0; normalise each feature element across batch. dim=1; normalise each batch element across features.
        else:
            assert False, f'Normalisation method {self.normalise_embedding} not implemented.'
        
        if self.use_dropout:
            encoding = F.dropout(encoding, p=self.p_dropout, training=self.training)

        return encoding
    
    def configure_optimizers(self):
        if self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            assert False, f'Optimizer {self.optimizer_name} not implemented.'

        if self.use_lr_scheduler:
            print('Using ReduceLROnPlateau scheduler.')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                             factor=0.1, patience=10, verbose=True)
            return {"optimizer": self.optimizer, 
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        else:
            return self.optimizer
    
    def pecl_pass(self, batch):
        '''Train the encoding model using the PECL method.'''
        im, pres_vec = batch
        
        # Forward pass
        im_enc = self.forward(im)
        if self.normalise_embedding == 'l2':
            pres_vec = F.normalize(pres_vec, p=2, dim=1)

        '''
        Maybe this should be split up.. Forward() should do the im_enc & pred.
        If pred not needed, forward() just doesnt do it and returns None. 

        Then this function just becomes the CL loss function.
        Another becomes loss for prediction. 

        Then train/val/test step just calls forward() and the loss function.
        
        '''
        loss, im_enc = self.pecl_loss(im_enc, pres_vec)
        return loss, None, im_enc

    def pecl_loss(self, im_enc, pres_vec):
        if self.pecl_distance_metric == 'softmax':
            if self.pecl_knn is not None:
                flatten_dist = False
            else:
                flatten_dist = True
            dist_array_ims = normalised_softmax_distance_batch(im_enc, flatten=flatten_dist)
            dist_array_pres = normalised_softmax_distance_batch(pres_vec, flatten=flatten_dist, knn=self.pecl_knn,
                                                                knn_hard_labels=self.pecl_knn_hard_labels)
                                                                
            # return dist_array_ims, dist_array_pres
            inds_one = torch.where(dist_array_pres > 0)
            dist_array_ims = dist_array_ims[inds_one]
            dist_array_pres = dist_array_pres[inds_one]

            ## cross entropy loss
            assert (dist_array_ims >= 0).all(), (dist_array_ims, im_enc)
            assert (dist_array_ims <= 1).all(), (dist_array_ims, im_enc)
            assert (dist_array_pres >= 0).all(), (dist_array_pres, pres_vec)
            assert (dist_array_pres <= 1).all(), (dist_array_pres, pres_vec)
            loss = (-1 * torch.log(dist_array_ims + 1e-8) * dist_array_pres).mean()
            assert loss >= 0, loss
        else:
            assert False, f'Distance metric {self.pecl_distance_metric} deprecated.'
            dist_array_ims = calculate_similarity_batch(im_enc, distance_metric=self.pecl_distance_metric)
            dist_array_pres = calculate_similarity_batch(pres_vec, distance_metric=self.pecl_distance_metric)
            
            loss_array = torch.abs(dist_array_ims - dist_array_pres)  # L1 loss
            loss = torch.mean(loss_array)

        return loss, im_enc

    def pred_pass(self, batch):
        '''Only trains the prediction model, not the encoding model.'''
        im, pres_vec = batch
        
        # Forward pass
        if self.train_im_enc_during_pred:
            im_enc = self.forward(im)
        else:
            with torch.no_grad():  # Don't train encoding model here. 
                im_enc = self.forward(im)
        pres_pred = self.prediction_model(im_enc)
        loss = self.pred_loss(pres_pred, pres_vec)
        return loss, None, im_enc
    
    def pred_and_pecl_pass(self, batch):
        '''Train the encoding model using the PECL method, and the prediction model.'''
        im, pres_vec = batch
        
        # Forward pass
        im_enc = self.forward(im)
        if self.normalise_embedding == 'l2':
            pres_vec_pecl = F.normalize(pres_vec, p=2, dim=1)
        else:
            assert False, f'Normalisation method {self.normalise_embedding} not implemented.'

        loss_pecl, im_enc = self.pecl_loss(im_enc, pres_vec_pecl)
        pres_pred = self.prediction_model(im_enc)
        loss_pred = self.pred_loss(pres_pred, pres_vec)

        loss = loss_pred + self.alpha_ratio_loss * loss_pecl
        return loss, (loss_pred, self.alpha_ratio_loss * loss_pecl), im_enc

    def pred_loss(self, pres_pred, pres_vec):
        if self.pred_train_loss == 'mse':
            loss = F.mse_loss(pres_pred, pres_vec)
        elif self.pred_train_loss == 'mae':
            loss = nn.L1Loss()(pres_pred, pres_vec)
        elif self.pred_train_loss == 'ce':
            loss = nn.CrossEntropyLoss(reduction='mean')(pres_pred, pres_vec)
        elif self.pred_train_loss == 'bce':
            loss = nn.BCELoss(reduction='mean')(pres_pred, pres_vec)
        elif self.pred_train_loss == 'weighted-ce' and self.class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')(pres_pred, pres_vec)
        elif self.pred_train_loss == 'weighted-bce' and self.class_weights is not None:
            loss = self.weighted_bce_loss(pres_pred, pres_vec)
        else:
            assert False, f'Prediction training loss {self.pred_train_loss} not implemented.'
        return loss
                
    def training_step(self, batch, batch_idx):
        loss, _, __ = self.forward_pass(batch)
        self.log(f'train_{self.name_train_loss}_loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, split_loss, im_enc = self.forward_pass(batch)
            self.log(f'val_{self.name_train_loss}_loss', loss, on_epoch=True, on_step=False)  # saving name loss function used so it can be recovered later
            self.log('val_loss', loss, on_epoch=True, on_step=False)  # also save as val_loss for tensorboard (and lr_scheduler etc)

            ## Evaluate split loss:
            if split_loss is not None:
                loss_pred, loss_pecl = split_loss
                self.log(f'val_pecl-{self.pecl_distance_metric}_loss', loss_pecl, on_epoch=True, on_step=False)
                ratio = loss_pred / loss_pecl
                self.log(f'val_ratio_pred_pecl', ratio, on_epoch=True, on_step=False)

            ## Evaluate prediction:
            pres_pred = self.prediction_model(im_enc)  ## not ideal to do this here, but for now it's fine.
            pres_vec = batch[1]
            assert pres_pred.shape == pres_vec.shape, f'Shape pred {pres_pred.shape}, shape labels {pres_vec.shape}'
            distance_pred_label_means = torch.mean(torch.abs(pres_vec.mean(0) - pres_pred.mean(0)))
            self.log(f'val_distance_pred_label_means', distance_pred_label_means, on_epoch=True, on_step=False)
            for k in [1, 5, 10, 20]:
                top_k_acc = self.top_k_accuracy(preds=pres_pred, target=pres_vec, k=k)
                self.log(f'val_top_{k}_acc', top_k_acc, on_epoch=True, on_step=False)
            mse_loss = F.mse_loss(pres_pred, pres_vec)
            self.log(f'val_mse_loss', mse_loss, on_epoch=True, on_step=False)
            mae_loss = nn.L1Loss()(pres_pred, pres_vec)
            self.log(f'val_mae_loss', mae_loss, on_epoch=True, on_step=False)
            ce_loss = nn.CrossEntropyLoss(reduction='mean')(pres_pred, pres_vec)
            self.log(f'val_ce_loss', ce_loss, on_epoch=True, on_step=False)
            bce_loss = nn.BCELoss(reduction='mean')(pres_pred, pres_vec)
            self.log(f'val_bce_loss', bce_loss, on_epoch=True, on_step=False)
            if self.class_weights is not None:
                bce_weighted_loss = self.weighted_bce_loss(pres_pred, pres_vec)
                self.log(f'val_weighted-bce_loss', bce_weighted_loss, on_epoch=True, on_step=False)
                ce_weighted_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')(pres_pred, pres_vec)
                self.log(f'val_weighted-ce_loss', ce_weighted_loss, on_epoch=True, on_step=False)
            return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, split_loss, im_enc = self.forward_pass(batch)
            self.log(f'test_{self.name_train_loss}_loss', loss)  # saving name loss function used so it can be recovered later
            self.log('test_loss', loss)  # also save as test_loss for tensorboard (and lr_scheduler etc)

            ## Evaluate split loss:
            if split_loss is not None:
                loss_pred, loss_pecl = split_loss
                self.log(f'test_pecl-{self.pecl_distance_metric}_loss', loss_pecl)
                ratio = loss_pred / loss_pecl
                self.log(f'test_ratio_pred_pecl', ratio)

            ## Evaluate prediction:
            pres_pred = self.prediction_model(im_enc)  ## not ideal to do this here, but for now it's fine.
            pres_vec = batch[1]
            assert pres_pred.shape == pres_vec.shape, f'Shape pred {pres_pred.shape}, shape labels {pres_vec.shape}'
            distance_pred_label_means = torch.mean(torch.abs(pres_vec.mean(0) - pres_pred.mean(0)))
            self.log(f'test_distance_pred_label_means', distance_pred_label_means)
            for k in [1, 5, 10, 20]:
                top_k_acc = self.top_k_accuracy(preds=pres_pred, target=pres_vec, k=k)
                self.log(f'test_top_{k}_acc', top_k_acc)
            mse_loss = F.mse_loss(pres_pred, pres_vec)
            self.log(f'test_mse_loss', mse_loss)
            mae_loss = nn.L1Loss()(pres_pred, pres_vec)
            self.log(f'test_mae_loss', mae_loss)
            ce_loss = nn.CrossEntropyLoss(reduction='mean')(pres_pred, pres_vec)
            self.log(f'test_ce_loss', ce_loss)
            bce_loss = nn.BCELoss(reduction='mean')(pres_pred, pres_vec)
            self.log(f'test_bce_loss', bce_loss)
            if self.class_weights is not None:
                bce_weighted_loss = self.weighted_bce_loss(pres_pred, pres_vec)
                self.log(f'test_weighted-bce_loss', bce_weighted_loss)
                ce_weighted_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')(pres_pred, pres_vec)
                self.log(f'test_weighted-ce_loss', ce_weighted_loss)
            return loss
    
    def weighted_bce_loss(self, preds, target):
        '''Weighted binary cross entropy loss.'''
        assert preds.shape == target.shape
        intermediate_loss = nn.BCELoss(reduction='none')(preds, target)  #TODO: add mean over just batch dimension here?
        # assert intermediate_loss.shape == target.shape, (intermediate_loss.shape, target.shape)
        assert intermediate_loss.shape[1] == self.n_species, (intermediate_loss.shape, self.n_species)
        intermediate_loss = torch.mean(intermediate_loss, dim=0)  
        assert intermediate_loss.shape == self.class_weights.shape, (intermediate_loss.shape, self.class_weights.shape)
        loss = torch.mean(intermediate_loss * self.class_weights)
        return loss

    def top_k_accuracy(self, preds, target, k=1):
        '''Calculate top-k accuracy: the proportion of samples where the target class is within the top k predicted classes.'''
        assert preds.shape == target.shape
        inds_sorted_preds = torch.argsort(preds, dim=1, descending=True)  # dim =1; sort along 2nd dimension (ie per sample)
        inds_sorted_target = torch.argsort(target, dim=1, descending=True)
        len_batch = preds.shape[0]
        
        ## Calculate top-k accuracy using tmp binary vectors that are 1 for the top-k predictions
        tmp_pred_greater_th = torch.zeros_like(preds)
        tmp_target_greater_th = torch.zeros_like(target)
        for row in range(len_batch):
            tmp_pred_greater_th[row, inds_sorted_preds[row, :k]] = 1
            tmp_target_greater_th[row, inds_sorted_target[row, :k]] = 1

        assert tmp_pred_greater_th.sum() <= k * len_batch, tmp_pred_greater_th.sum() 
        assert tmp_target_greater_th.sum() <= k * len_batch, tmp_target_greater_th.sum()

        tmp_joint = tmp_pred_greater_th * tmp_target_greater_th
        n_present = torch.sum(tmp_joint, dim=1)  ## sum per batch sample

        for n in n_present:
            assert n <= k, n_present

        top_k_acc = n_present.float() / k  # accuracy per batch sample 
        top_k_acc = top_k_acc.mean()

        return top_k_acc
    
    def store_metrics(self, metrics):
        if self.df_metrics is not None:
            print('-- When saving metrics, df_metrics already exists. Appending to old_df_metrics.')
            if hasattr(self, 'old_df_metrics'):
                assert type(self.old_df_metrics) == list
                self.old_df_metrics.append(copy.deepcopy(self.df_metrics))
            else:
                self.old_df_metrics = [copy.deepcopy(self.df_metrics)]
            print(f'-- Old df_metrics stored in old_df_metrics list of length {len(self.old_df_metrics)}')

        self.n_epochs_converged = len(metrics) 
        metrics_float = []
        self.set_metric_names = set()
        for ii in range(len(metrics)):
            metrics_float.append({})
            for key in metrics[ii].keys():
                self.set_metric_names.add(key)
                value = metrics[ii][key]
                if type(value) == float:
                    continue 
                if type(value) == torch.Tensor:
                    metrics_float[ii][key] = value.detach().cpu().numpy()
                assert type(metrics_float[ii][key]) == np.ndarray, type(metrics_float[ii][key])
                assert metrics_float[ii][key].shape == (), metrics_float[ii][key]
                metrics_float[ii][key] = float(metrics_float[ii][key])

        self.metric_arrays = {}
        for key in self.set_metric_names:
            self.metric_arrays[key] = np.zeros(self.n_epochs_converged) + np.nan
            for ii in range(self.n_epochs_converged):
                if key in metrics_float[ii]:
                    self.metric_arrays[key][ii] = metrics_float[ii][key]
         
        self.df_metrics = pd.DataFrame(self.metric_arrays)
        return 

    def save_stats(self, folder='/Users/t.vanderplas/models/PECL/stats/',
                   verbose=1):
        '''Save logger stats & model params only. '''
        assert self.df_metrics is not None, 'Metrics not stored yet.'
        assert os.path.exists(folder), f'Folder {folder} does not exist.'
        ## Save v_num that is used for tensorboard
        self.v_num = self.logger.version
        ## Save logging directory that is used for tensorboard
        self.log_dir = self.logger.log_dir

        if self.model_name is None:
            timestamp = cdu.create_timestamp()
            self.model_name = f'PECL-ImEn_{timestamp}_vnum-{self.v_num}'

        self.dict_save = {
            'hparams': {**self.hparams, **{'name_train_loss': self.name_train_loss},
                        'pred_train_loss': self.pred_train_loss,
                        'n_epochs_converged': self.n_epochs_converged},
            'v_num': self.v_num,
            'log_dir': self.log_dir,
            'logger': self.logger.__dict__,
            'df_metrics': self.df_metrics
        }
        self.filename = f'{self.model_name}_stats.pkl'
        self.filepath = os.path.join(folder, self.filename)
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.dict_save, f)
        if verbose > 0:
            print(f'Stats saved as {self.filename} at {self.filepath}')

    def save_model(self, folder='/Users/t.vanderplas/models/PECL/full_models/', 
                   verbose=1):
        '''Save model'''
        assert self.df_metrics is not None, 'Metrics not stored yet.' 
        ## Save v_num that is used for tensorboard
        self.v_num = self.logger.version
        ## Save logging directory that is used for tensorboard
        self.log_dir = self.logger.log_dir
        
        if self.model_name is None:
            timestamp = cdu.create_timestamp()
            self.model_name = f'PECL-ImEn_{timestamp}_vnum-{self.v_num}'
        self.filename = f'{self.model_name}.data'
        self.filepath = os.path.join(folder, self.filename)

        file_handle = open(self.filepath, 'wb')
        pickle.dump(self, file_handle)

        if verbose > 0:
            print(f'PECL-ImEn model saved as {self.filename} at {self.filepath}')
        return self.filepath

def load_model(folder='/Users/t.vanderplas/models/PECL/full_models/', 
               filename='', verbose=1):
    '''Load previously saved (pickled) model'''
    assert filename != '', 'Filename not provided.'
    assert filename.endswith('.data'), f'Filename {filename} should end with .data'
    with open(os.path.join(folder, filename), 'rb') as f:
        model = pickle.load(f)

    if verbose > 0:  
        print(f'Loaded {model}')
        if hasattr(model, 'description') and verbose > 0:
            print(model.description)

    return model 

def load_model_from_ckpt(v_num=None, filepath=None, base_folder='/Users/t.vanderplas/models/PECL/lightning_logs/'):
    '''Load model from checkpoint file.'''
    assert filepath is not None or v_num is not None, 'Version number and filepath not provided.'
    if filepath is None:
        assert v_num is not None, 'Version number and filepath not provided.'
        assert type(v_num) == int, f'Expected int for version number, but got {type(v_num)}'
        folder_v_num = os.path.join(base_folder, f'version_{v_num}')
        assert os.path.exists(folder_v_num), f'Folder {folder_v_num} does not exist.'
        folder_ckpt = os.path.join(folder_v_num, 'checkpoints')
        contents_folder_ckpt = os.listdir(folder_ckpt)
        assert len(contents_folder_ckpt) == 1, f'Expected 1 file in folder {folder_ckpt}, but got {len(contents_folder_ckpt)}'
        filepath = os.path.join(folder_ckpt, contents_folder_ckpt[0])
    elif v_num is None:
        assert filepath is not None, 'Version number and filepath not provided.'
        assert filepath.endswith('.ckpt'), f'Filepath {filepath} should end with .ckpt'
        assert os.path.exists(filepath), f'File {filepath} does not exist.'
    # model = ImageEncoder.load_from_checkpoint(filepath)
    # return model
    assert False, 'Not implemented yet.'

def load_stats(folder='/Users/t.vanderplas/models/PECL/stats/', filename=None, timestamp=None, verbose=1):
    '''Load previously saved (pickled) stats'''
    assert (filename is not None and timestamp is None) or (filename is None and timestamp is not None), 'Provide either filename or timestamp, not both.'
    if filename is None:
        list_files = os.listdir(folder)
        list_files = [f for f in list_files if f.endswith('.pkl')]
        list_files = [f for f in list_files if timestamp in f]
        assert len(list_files) == 1, f'Expected 1 file with timestamp {timestamp}, but got {len(list_files)}'
        filename = list_files[0]
    assert filename != '', 'Filename not provided.'
    assert filename.endswith('.pkl'), f'Filename {filename} should end with .pkl'
    with open(os.path.join(folder, filename), 'rb') as f:
        dict_load = pickle.load(f)
    if verbose > 0:
        print(f'Loaded stats from {filename} at {folder}')
    return dict_load

def normalised_softmax_distance_batch(samples, temperature=0.5, exclude_diag_in_denominator=True,
                                      flatten=True, knn=None, knn_hard_labels=False,
                                      similarity_function='inner', inner_prod_only=False):
    '''Calculate the distance between two embeddings using the normalised softmax distance.'''
    assert temperature > 0, f'Temperature {temperature} should be > 0.'
    assert samples.ndim == 2, f'Expected 2D tensor, but got {samples.ndim}D tensor.'  # (batch, features)
    assert (samples <= 1).all(), f'Values should be <= 1, but got {samples.max()}'
    if similarity_function == 'inner':
        pass
    elif similarity_function == 'cosine':
        samples = F.normalize(samples, p=2, dim=1)
        assert False, 'Expected samples to be already normalised..'
    else:
        assert False, f'Similarity function {similarity_function} not implemented.'

    inner_prod_mat = torch.mm(samples, samples.t())
    if inner_prod_only is False:
        inner_prod_mat = inner_prod_mat / temperature
        inner_prod_mat = torch.exp(inner_prod_mat)
        sum_inner_prod_mat = torch.sum(inner_prod_mat, dim=1)
        if exclude_diag_in_denominator:
            diag = torch.diag(inner_prod_mat)
            sum_inner_prod_mat = sum_inner_prod_mat - diag
        sum_inner_prod_mat = sum_inner_prod_mat + 1e-8  # avoid division by zero
        inner_prod_mat = inner_prod_mat / sum_inner_prod_mat[:, None]

    if knn is not None:
        assert flatten is False, 'Flatten should be False if KNN is used.'
        assert type(knn) == int, f'Expected int for knn, but got {type(knn)}'
        assert knn > 0, f'Expected knn > 0, but got {knn}'
        assert knn < samples.shape[0] - 1, f'Expected knn < number of samples - 1, but got {knn} and {samples.shape[0]}'
        inner_prod_mat = inner_prod_mat - torch.diag(inner_prod_mat.diag())  # set diagonal to 0 so it doesn't get picked with KNN
        knn_inner_prod_mat = torch.zeros_like(inner_prod_mat)
        inds_positive = torch.topk(inner_prod_mat, k=knn, dim=1, largest=True, sorted=False)[1]
        if knn_hard_labels:
            knn_inner_prod_mat.scatter_(1, inds_positive, 1)
        else:
            for row, cols in enumerate(inds_positive):
                knn_inner_prod_mat[row, cols] = inner_prod_mat[row, cols]
        return knn_inner_prod_mat
    if flatten:  # only return upper triangular part because of symmetry
        inds_upper_triu = torch.triu_indices(inner_prod_mat.shape[0], inner_prod_mat.shape[1], offset=1)
        inner_prod_mat = inner_prod_mat[inds_upper_triu[0], inds_upper_triu[1]]
    return inner_prod_mat
    
def calculate_similarity_batch(samples, distance_metric='cosine'):
    curr_batch_size = len(samples)
    ## Calculate distance between pairs of images and pairs of presence vectors
    similarity_array_samples = torch.zeros(curr_batch_size * (curr_batch_size - 1) // 2)
    ind_pair = -1
    for i in range(curr_batch_size):
        for j in range(i + 1, curr_batch_size):  # avoid duplicates
            ind_pair += 1
            # print(ind_pair, i, j, samples[i].shape, samples[j].shape)
            if distance_metric == 'cosine':
                similarity_array_samples[ind_pair] = 1 - F.cosine_similarity(samples[i], samples[j], dim=0)
            elif distance_metric == 'euclidean':
                similarity_array_samples[ind_pair] = F.pairwise_distance(samples[i], samples[j], p=2)
            else:
                assert False, f'Distance metric {distance_metric} not implemented.'
    return similarity_array_samples

def train_pecl(model=None, freeze_resnet_fc_loaded_model=False,
               n_enc_channels=256, 
               n_layers_mlp_resnet=1, n_layers_mlp_pred=2,
               pretrained_resnet='seco', freeze_resnet=True,
               resnet_version=18, pecl_distance_metric='softmax',
               pred_train_loss='bce', use_class_weights=False,
               normalise_embedding='l2', n_bands=4,
               training_method='pecl', lr=1e-3, batch_size=64, n_epochs_max=20, 
               image_folder=None, presence_csv=None,  # None will use default paths 
               species_process='all', p_dropout=0,
               pecl_knn=5, pecl_knn_hard_labels=False, alpha_ratio_loss=0.01,
               use_lr_scheduler=False, stop_early=False,
               verbose=1, fix_seed=42, use_mps=True,
               filepath_train_val_split=None,
               save_model=False, save_stats=True):
    assert filepath_train_val_split is not None, 'Expecting filepath_train_val_split to be set.'
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
        acc_use = 'gpu'
        # acc_use = 'cpu'
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
                              shuffle_order_data=True, species_process=species_process,
                              augment_image=True, 
                              n_bands=n_bands, zscore_im=True, mode='train')
    train_ds, val_ds, test_ds = ds.split_into_train_val(filepath=filepath_train_val_split)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus, 
                          shuffle=True, persistent_workers=True) #drop_last=True, pin_memory=True
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=n_cpus, 
                        shuffle=False,  persistent_workers=True) 
    if test_ds is not None:
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=n_cpus, 
                             shuffle=False,  persistent_workers=True)
    else:
        test_dl = None

    if model is None:
        time_created = cdu.create_timestamp(include_seconds=True)
        model = ImageEncoder(n_species=ds.n_species, n_enc_channels=n_enc_channels, 
                             n_layers_mlp_resnet=n_layers_mlp_resnet, n_layers_mlp_pred=n_layers_mlp_pred,
                             pred_train_loss=pred_train_loss, 
                            pretrained_resnet=pretrained_resnet, freeze_resnet=freeze_resnet,
                            optimizer_name='Adam', resnet_version=resnet_version,
                            class_weights=ds.weights_values if use_class_weights else None,
                            pecl_distance_metric=pecl_distance_metric,
                            normalise_embedding=normalise_embedding,
                            pecl_knn=pecl_knn, pecl_knn_hard_labels=pecl_knn_hard_labels,
                            lr=lr, n_bands=n_bands, use_mps=use_mps,
                            use_lr_scheduler=use_lr_scheduler,
                            training_method=training_method,
                            alpha_ratio_loss=alpha_ratio_loss,
                            p_dropout=p_dropout,
                            time_created=time_created, batch_size_used=batch_size,
                            verbose=verbose, seed_used=fix_seed)
    else:
        assert type(model) == ImageEncoder, f'Expected model to be ImageEncoder, but got {type(model)}'
        assert model.n_species == ds.n_species, f'Number of species in model {model.n_species} does not match number of species in dataset {ds.n_species}.'
        assert model.n_enc_channels == n_enc_channels, f'Number of encoding channels in model {model.n_enc_channels} does not match number of encoding channels in dataset {n_enc_channels}.'
        assert model.n_layers_mlp_pred == n_layers_mlp_pred, f'Number of layers in prediction model {model.n_layers_mlp_pred} does not match number of layers in dataset {n_layers_mlp_pred}.'
        assert model.n_layers_mlp_resnet == n_layers_mlp_resnet, f'Number of layers in resnet model {model.n_layers_mlp_resnet} does not match number of layers in dataset {n_layers_mlp_resnet}.'
        assert model.resnet_version == resnet_version, f'Resnet version in model {model.resnet_version} does not match resnet version in dataset {resnet_version}.'
        assert model.normalise_embedding == normalise_embedding, f'Normalisation method in model {model.normalise_embedding} does not match normalisation method in dataset {normalise_embedding}.'
        assert model.n_bands == n_bands, f'Number of bands in model {model.n_bands} does not match number of bands in dataset {n_bands}.'

        model.lr = lr
        model.freeze_resnet = freeze_resnet
        model.pecl_distance_metric = pecl_distance_metric
        model.pecl_knn = pecl_knn
        model.pecl_knn_hard_labels = pecl_knn_hard_labels
        model.use_lr_scheduler = use_lr_scheduler
        model.verbose = verbose
        model.seed_used = fix_seed
        model.model_name = None  # reset model name
        model.batch_size_used = batch_size

        model.build_class_weights(class_weights=ds.weights_values if use_class_weights else None)
        model.build_training_method(training_method=training_method, pred_train_loss=pred_train_loss)
       
        if freeze_resnet_fc_loaded_model:  # freezing resnet fc head. So needs to be tuned already (hence only for loaded model).
            model.freeze_resnet_layers(freeze_all_but_last=freeze_resnet, freeze_last=True)

    cb_metrics = MetricsCallback()
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                            filename="best_checkpoint_val-{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}"),
                 cb_metrics]
    if stop_early:
        callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'))

    trainer = pl.Trainer(max_epochs=n_epochs_max, accelerator=acc_use,
                         log_every_n_steps=5,  # train loss logging steps (each step = 1 batch)
                         reload_dataloaders_every_n_epochs=1, # reload such that train_dl re-shuffles.  https://github.com/Lightning-AI/pytorch-lightning/discussions/7332
                         callbacks=callbacks, logger=tb_logger)

    timestamp_start = datetime.datetime.now()
    print(f'-- Starting training at {timestamp_start} with {n_epochs_max} epochs.')
    trainer.fit(model, train_dl, val_dl)

    timestamp_end = datetime.datetime.now()
    print(f'-- Finished training at {timestamp_end}.')

    if test_dl is not None:
        trainer.test(test_dataloaders=test_dl)

    model.store_metrics(metrics=cb_metrics.metrics)
    if save_stats:
        model.save_stats(verbose=1)
    if save_model:
        model.save_model(verbose=1)
    return model, (train_dl, val_dl)

class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback.
    https://lightning.ai/forums/t/how-to-access-the-logged-results-such-as-losses/155
    """

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)

class MeanRates(ImageEncoder):
    def __init__(self, train_ds, val_ds):
        super().__init__(training_method='pred',
                         class_weights=None,
                         pred_train_loss='bce')
        
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.calculate_mean_rates()
        self.evaluate_val_losses()

    def calculate_mean_rates(self):
        print('Calculating mean rates for training and validation datasets.')
        all_labels = []
        for sample in tqdm(self.train_ds):
            all_labels.append(sample[1][None, :])    
           
        all_labels = torch.cat(all_labels, dim=0)
        print(f'All labels shape: {all_labels.shape}')
        mean_rates = all_labels.mean(dim=0)
        self.train_mean_rates = mean_rates
        assert self.train_mean_rates.ndim == 1, self.train_mean_rates.shape
        assert self.train_mean_rates.shape[0] == self.n_species, (self.train_mean_rates.shape, self.n_species)
        assert (self.train_mean_rates >= 0).all(), self.train_mean_rates
        assert (self.train_mean_rates <= 1).all(), self.train_mean_rates

    def evaluate_val_losses(self):
        print('Evaluating validation losses for validation dataset.')
        val_loss_dict = {x: [] for x in ['mae', 'mse', 'bce', 'top_5', 'top_10', 'top_20']}
        all_val_labels = []
        for sample in tqdm(self.val_ds):
            all_val_labels.append(sample[1][None, :])
        pres_vec = torch.cat(all_val_labels, dim=0)
        len_batch = pres_vec.shape[0]

        pred_rates = self.train_mean_rates[None, :].repeat(len_batch, 1)
        assert pred_rates.shape == pres_vec.shape, (pred_rates.shape, pres_vec.shape)

        mae_loss = nn.L1Loss()(pred_rates, pres_vec)
        val_loss_dict['mae'].append(mae_loss.item())
        mse_loss = F.mse_loss(pred_rates, pres_vec)
        val_loss_dict['mse'].append(mse_loss.item())
        bce_loss = nn.BCELoss(reduction='mean')(pred_rates, pres_vec)
        val_loss_dict['bce'].append(bce_loss.item())
        for k in [5, 10, 20]:
            top_k_acc = self.top_k_accuracy(preds=pred_rates, target=pres_vec, k=k)
            val_loss_dict[f'top_{k}'].append(top_k_acc.item())

        self.val_loss_dict = val_loss_dict
        print(f'Validation losses: {val_loss_dict}')