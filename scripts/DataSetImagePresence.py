
import numpy as np 
import pandas as pd
import os, copy, ast 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA 
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit
from geopy.distance import distance as geodist # avoid naming confusion
import torch
from torchvision.transforms import v2
import rasterio
import rasterio.features, rasterio.plot
import rioxarray as rxr
import xarray as xr
import rasterio
from tqdm import tqdm
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()
import create_dataset_utils as cdu 
import torch.utils.data

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
                 mode='train', use_testing_data=False,
                 augment_image=True, verbose=1, dataset_name='s2bms'):
        # super(DataSetImagePresence, self).__init__()
        self.image_folder = image_folder
        self.presence_csv = presence_csv
        self.mode = mode
        self.verbose = verbose
        self.zscore_im = zscore_im
        self.use_testing_data = use_testing_data
        self.dataset_name = dataset_name
        assert self.dataset_name in ['s2bms', 'satbird-kenya', 'satbird-usawinter'], f'Dataset name {self.dataset_name} not implemented.'
        if self.zscore_im:
            if self.dataset_name == 's2bms':  ## Values obtained from full data set (1336 images)
                self.norm_means = np.array([661.1047,  770.6800,  531.8330, 3228.5588]).astype(np.float32) 
                self.norm_std = np.array([640.2482,  571.8545,  597.3570, 1200.7518]).astype(np.float32) 
            elif self.dataset_name == 'satbird-kenya':  ## From SatBird stats/means_rgbnir.npy (and stds_)
                self.norm_means = np.array([1905.25581818, 1700.55621907, 1554.70146535, 2432.34535847]).astype(np.float32) 
                self.norm_std = np.array([1147.77209359,  777.33364953,  506.84587793, 1632.70761804]).astype(np.float32) 
            elif self.dataset_name == 'satbird-usawinter':  ## From SatBird stats/means_rgbnir.npy (and stds_)
                self.norm_means = np.array([2344.2988485, 2253.80917105, 2124.48143172, 3197.92686188]).astype(np.float32) 
                self.norm_std = np.array([1739.14810048, 1714.90654424, 1763.818098, 1672.77914703]).astype(np.float32) 
            self.norm_means = self.norm_means[:, None, None]
            self.norm_std = self.norm_std[:, None, None]
        else:
            self.norm_means = None
            self.norm_std = None
        if self.dataset_name == 's2bms':
            self.prefix_name_loc = 'UKBMS_'   ## '''Expected format of name_loc in presence csv: UKBMS_loc-xxxxx, in image folder: prefix_UKBMS_loc-xxxxx_suffix1_suffix2.tif.'''
            self.cols_not_species = ['tuple_coords', 'n_visits', 'name_loc']
        elif 'satbird' in self.dataset_name:
            self.prefix_name_loc = ''
            self.cols_not_species = ['tuple_coords', 'n_visits', 'name_loc']
        self.augment_image = augment_image
        self.shuffle_order_data = shuffle_order_data
        self.species_process = species_process
        self.n_bands = n_bands
        self.load_data()

    def load_data(self):
        if self.use_testing_data:
            self.set_test_data_paths()
            
        assert os.path.exists(self.presence_csv), f"Presence csv does not exist: {self.presence_csv}"
        df_presence = pd.read_csv(self.presence_csv, index_col=0)
        locs_presence = [x.lstrip(self.prefix_name_loc) for x in df_presence['name_loc'].values]

        assert os.path.exists(self.image_folder), f"Image folder does not exist: {self.image_folder}"
        content_image_folder = os.listdir(self.image_folder)
        if self.dataset_name == 's2bms':
            locs_images = [x.split('_')[2] for x in content_image_folder]
            suffix_images = np.unique(['_'.join(x.split('_')[3:]) for x in content_image_folder])
            self.suffix_images = suffix_images
            prefix_images = np.unique([x.split('_')[0] for x in content_image_folder])
            assert len(prefix_images) == 1, "Multiple prefixes found in image folder."
            self.prefix_images = prefix_images[0]
        elif 'satbird' in self.dataset_name:
            locs_images = [x.strip('.tif') for x in content_image_folder]
            self.suffix_images = ['.tif']
            self.prefix_images = ''

        tmp_is_present = np.array([True if x in locs_images else False for x in locs_presence])
        if self.verbose:
            print(f'Found {np.sum(tmp_is_present)} out of {len(tmp_is_present)} images in the image folder.')

        df_presence = df_presence[df_presence['name_loc'].isin([f'{self.prefix_name_loc}{x}' for x in locs_images])]
        assert len(df_presence) == np.sum(tmp_is_present), "Mismatch between presence/absence data and image folder."
        if self.shuffle_order_data:
            print('Shuffling data.')
            df_presence = df_presence.sample(frac=1, replace=False)
        else:
            print('Sorting data by name_loc.')
            df_presence = df_presence.sort_values(by='name_loc')
        df_presence = df_presence.reset_index(drop=True)

        original_species_list = [x for x in df_presence.columns if x not in self.cols_not_species]
        n_original_species = len(original_species_list)
        if self.species_process == 'all':
            pass 
        elif self.species_process == 'priority_species' or self.species_process == 'priority_species_present':
            assert self.dataset_name == 's2bms', 'Priority species only implemented for UKBMS data.'
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
            cols_keep = self.cols_not_species + priority_species
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
            cols_keep = self.cols_not_species + [original_species_list[x] for x in cols_species_top20]
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

            df_presence = pd.concat([df_presence[self.cols_not_species], df_presence_pca], axis=1)
            total_expl_var = np.sum(pca.explained_variance_ratio_)
            print(f'PCA with {n_pcs_keep} components explains {100 * total_expl_var:.1f}% of the variance.')
        else:
            assert False, f'Species process {self.species_process} not implemented.'

        self.species_list = [x for x in df_presence.columns if x not in self.cols_not_species]
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

    def set_test_data_paths(self):
        '''Load mock data in the same format as load_data() for CI testing.'''
        mock_presence_csv = os.path.join(path_dict_pecl['repo'], 'tests/data_test/presence_tests/ukbms_presence_test16.csv')
        mock_image_folder = os.path.join(path_dict_pecl['repo'], 'tests/data_test/images_tests')
        self.presence_csv = mock_presence_csv
        self.image_folder = mock_image_folder
        print(f'Loading mock data from {mock_presence_csv} and {mock_image_folder}.')
        
    def find_image_path(self, name_loc):
        if len(self.suffix_images) == 1:
            if 'satbird' in self.dataset_name:
                im_file_name = f'{name_loc}{self.suffix_images[0]}'
            else:
                im_file_name = f'{self.prefix_images}_{name_loc}_{self.suffix_images[0]}'
            im_file_path = os.path.join(self.image_folder, im_file_name)
            if os.path.exists(im_file_path):
                return im_file_path
        else:
            for s in self.suffix_images:
                im_file_name = f'{self.prefix_images}_{name_loc}_{s}'
                im_file_path = os.path.join(self.image_folder, im_file_name)
                if os.path.exists(im_file_path):
                    return im_file_path
        return None

    def load_image(self, name_loc):
        im_file_path = self.find_image_path(name_loc=name_loc)
        assert im_file_path is not None, f'Image file for location {name_loc} not found.'
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
            im = v2.CenterCrop(224)(im)
        
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

        n_in_splits = len(split_indices['train_indices']) + len(split_indices['val_indices'])
        if 'test_indices' in split_indices:
            n_in_splits += len(split_indices['test_indices'])
        if n_in_splits != len(self):
            print(f'WARNING: Number of indices in splits {n_in_splits} does not match number of samples {len(self)}.')

        train_inds = np.where(self.df_presence['name_loc'].isin(split_indices['train_indices']))[0]
        val_inds = np.where(self.df_presence['name_loc'].isin(split_indices['val_indices']))[0]
        train_ds = torch.utils.data.Subset(self, train_inds)
        val_ds = torch.utils.data.Subset(self, val_inds)

        train_ds.dataset.mode = 'train'
        val_ds.dataset.mode = 'val'
        if 'test_indices' in split_indices.keys():
            test_inds = np.where(self.df_presence['name_loc'].isin(split_indices['test_indices']))[0]
            test_ds = torch.utils.data.Subset(self, test_inds)
            test_ds.dataset.mode = 'val'
        else:
            test_ds = None
        return train_ds, val_ds, test_ds
    
    def split_and_save(self, split_fun='spatial_clusters', create_test=True, save_indices=True):
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

                print(f'Created {len(train_inds)} train, {len(val_inds)} val, {len(test_inds)} test indices.')
                train_names = self.df_presence.iloc[train_inds]['name_loc']
                val_names = self.df_presence.iloc[val_inds]['name_loc']
                test_names = self.df_presence.iloc[test_inds]['name_loc']
                split_indices = {'train_indices': train_names,
                                'val_indices': val_names,
                                'test_indices': test_names,
                                'clusters': clusters}
            else:
                ## Split into train and test:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                train_inds, test_inds = next(gss.split(np.arange(len(coords)), groups=clusters))

                clusters_train = clusters[train_inds]
                clusters_test = clusters[test_inds]
                assert len(np.intersect1d(train_inds, test_inds)) == 0, np.intersect1d(train_inds, test_inds)
                assert len(np.intersect1d(clusters_train, clusters_test)) == 0, np.intersect1d(clusters_train, clusters_test)

                train_names = self.df_presence.iloc[train_inds]['name_loc']
                test_names = self.df_presence.iloc[test_inds]['name_loc']
                split_indices = {'train_indices': train_names,
                                  'val_indices': test_names,
                                  'clusters': clusters}
                print(f'Created {len(train_inds)} train, {len(test_inds)} val indices.')
        timestamp = cdu.create_timestamp()
        if save_indices:
            torch.save(split_indices, os.path.join(path_dict_pecl['repo'], f'content/split_indices_{self.dataset_name}_{timestamp}.pth'))
            print(f'Saved split indices to split_indices_{timestamp}.pth')
        return split_indices
