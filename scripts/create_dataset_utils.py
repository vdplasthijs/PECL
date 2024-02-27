## Author: Thijs van der Plas

import os, sys 
import numpy as np 
import pandas as pd 
from dwca.read import DwCAReader
import matplotlib.pyplot as plt
import seaborn as sns 
import shapely 
from shapely.geometry import Point, Polygon
import datetime
# import warnings
# from shapely.errors import ShapelyDeprecationWarning
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
import geopandas as gpd
# import h3pandas 
from tqdm import tqdm
import scipy 
from matplotlib.colors import ListedColormap
import scipy.spatial
import scipy.cluster
import json

import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()

sys.path.append(os.path.join(path_dict_pecl['home'], 'repos/cnn-land-cover/scripts/'))
import land_cover_analysis as lca 
import land_cover_visualisation as lcv
import land_cover_models as lcm 
import ee, geemap 
sys.path.append(os.path.join(path_dict_pecl['repo'], 'content/'))
import api_keys
sys.path.append(os.path.join(path_dict_pecl['home'], 'repos/reproducible_figures/scripts/'))
import rep_fig_vis as rfv

ee.Authenticate()
ee.Initialize(project=api_keys.GEE_API)
geemap.ee_initialize()


def load_df_gbif(path_gbif_ds=None, verbose=1):
    if path_gbif_ds is None:
        path_gbif_ds = path_dict_pecl['ukbms_full_dataset'] 
    with DwCAReader(path_gbif_ds) as dwca:
        df = dwca.pd_read('occurrence.txt', parse_dates=True)

    # df = df[df['institutionCode'] == 'UKBMS']
        
    if verbose:
        print('Loaded GBIF dataset with {} records'.format(len(df)))
    return df

def add_tuple_coords(df):
    df['tuple_coords'] = [(x, y) for x, y in zip(df.decimalLongitude, df.decimalLatitude)]
    return df

def create_names_unique_locs(df, 
                             save_json=False, path_save=None):
    df = add_tuple_coords(df)
    df_unique_pairs = df[['tuple_coords', 'footprintWKT']].value_counts().reset_index(name='count')
    ## drop duplicates:
    df_unique_pairs.drop_duplicates(subset=['tuple_coords'], inplace=True, keep=False)
    df_unique_pairs.drop_duplicates(subset=['footprintWKT'], inplace=True, keep=False)
    df_unique_pairs['polygon'] = [shapely.wkt.loads(wkt) for wkt in df_unique_pairs.footprintWKT]
    n_digits_unique = len(str(len(df_unique_pairs)))
    df_unique_pairs['name_loc'] = [f'UKBMS_loc-{str(i).zfill(n_digits_unique)}' for i in range(len(df_unique_pairs))]

    if save_json:
        if path_save is None:
            path_save = os.path.join(path_dict_pecl['repo'], 'content/df_mapping_locs.json')
        df_unique_pairs.to_json(path_save)
    return df_unique_pairs
    
def load_names_unique_locs(df=None, path_json=None):
    if df is not None:
        df = add_tuple_coords(df)
    if path_json is None:
        path_json = os.path.join(path_dict_pecl['repo'], 'content/df_mapping_locs.json')
    df_unique_pairs = pd.read_json(path_json)
    df_unique_pairs['tuple_coords'] = df_unique_pairs.tuple_coords.apply(lambda x: (x[0], x[1]))  # convert to tuple from list s
    return df_unique_pairs, df
    
def count_rows_per_unique_val(df, col_interest='footprintWKT'):
    # unique_vals = df[col_interest].unique()
    count_obs_per_point = df.groupby(col_interest).size()
    return count_obs_per_point

def plot_obs_per_location(df, col_interest='footprintWKT'):
    count_obs_per_point = count_rows_per_unique_val(df, col_interest=col_interest)
    print(f'Unique number of points {len(count_obs_per_point)} out of {len(df)}')

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].hist(count_obs_per_point.sort_values(), bins=50)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Number of records per location')
    ax[0].set_ylabel('Nu mber of locations')

    arr_threshold = np.array([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000])
    arr_n_obs_greater = np.array([np.sum(count_obs_per_point >= t) for t in arr_threshold])
    ax[1].plot(arr_threshold, arr_n_obs_greater, 'o-')
    for i, t in enumerate(arr_threshold):
        n_locs = arr_n_obs_greater[i]
        ax[1].plot([t, t], [0, n_locs], 'k--', alpha=0.5)
        ax[1].plot([0, t], [n_locs, n_locs], 'k--', alpha=0.5)

    ax[1].set_xlabel('Minimum number of records per location')
    ax[1].set_ylabel('Number of locations with at least this number of records')

    arr_n_obs_total = np.zeros(len(arr_threshold))
    # arr_n_obs_greater = np.array([np.sum(count_obs_per_point >= t) for t in arr_threshold])
    for i, t in tqdm(enumerate(arr_threshold)):
        df_filtered = df[df[col_interest].isin(count_obs_per_point[count_obs_per_point >= t].index)]
        n_obs = len(df_filtered)
        arr_n_obs_total[i] = n_obs
    ax[2].plot(arr_threshold, arr_n_obs_total, 'o-')

def clean_bms_data(df, threshold_n_obs_per_location=200, col_location='footprintWKT',
                    verbose=1):
    count_obs_per_point = count_rows_per_unique_val(df, col_interest=col_location)
    df_clean = df[df[col_location].isin(count_obs_per_point[count_obs_per_point >= threshold_n_obs_per_location].index)]
    if verbose:
        print(f'Kept {len(df_clean)} records at locations with at least {threshold_n_obs_per_location} observations')

    unique_locations = df_clean[col_location].unique()
    if verbose:
        print(f'Unique number of locations {len(unique_locations)} out of {len(df)}')

    df_clean = df_clean.dropna(subset=['decimalLatitude', 'decimalLongitude'])
    cols_keep = ['license', 'institutionCode', 'decimalLatitude', 'decimalLongitude', 
                'coordinateUncertaintyInMeters', 'footprintWKT',
                'eventDate', 'year', 'month', 'day', 'verbatimLocality', 'species', 'speciesKey',
                'scientificName', 'datasetKey', 'locality', 'basisOfRecord']
    df_clean = df_clean[cols_keep]
    df_clean['tuple_coords'] = [(x, y) for x, y in zip(df_clean.decimalLongitude, df_clean.decimalLatitude)]
    df_clean['point'] = [Point(xy) for xy in zip(df_clean.decimalLongitude, df_clean.decimalLatitude)]
    df_clean['polygon'] = [shapely.wkt.loads(wkt) for wkt in df_clean.footprintWKT]
    df_clean['area'] = [p.area for p in df_clean.polygon]
    area_threshold = 0.0002  ## just a sanity check in case there are erroneous polygons
    n_obs = len(df_clean)
    df_clean = df_clean[df_clean.area < area_threshold]
    n_new_obs = len(df_clean)
    if n_new_obs != n_obs:
        print(f'Warning: {n_obs - n_new_obs} records were removed because of erroneous polygons')
    return df_clean

def create_minimal_df_bms(df_clean):
    unique_licenses = df_clean.license.unique()
    assert len(unique_licenses) == 1 and unique_licenses[0] == 'CC_BY_4_0', unique_licenses
    unique_inst_codes = df_clean.institutionCode.unique()
    assert len(unique_inst_codes) == 1 and unique_inst_codes[0] == 'UKBMS'
    unique_dataset_keys = df_clean.datasetKey.unique()
    assert len(unique_dataset_keys) == 1 
    unique_coord_uncertainties = df_clean.coordinateUncertaintyInMeters.unique()
    assert len(unique_coord_uncertainties) == 1 and unique_coord_uncertainties[0] == 707.1
    unique_basis_of_records = df_clean.basisOfRecord.unique()
    assert len(unique_basis_of_records) == 1 and unique_basis_of_records[0] == 'HUMAN_OBSERVATION'

    df_minimal = df_clean.drop(['license', 'institutionCode', 'coordinateUncertaintyInMeters', 'datasetKey', 
                            'verbatimLocality', 'scientificName', 'locality', 'basisOfRecord',
                            'area', 'point', 'decimalLongitude', 'decimalLatitude'], axis=1)
    return df_minimal

def create_species_abundance_per_loc_and_date(df_minimal, verbose=1):
    species_list = df_minimal.species.unique()
    if verbose:
        print(f'Unique number of species {len(species_list)}')

    ## add species as columns, default to 0, 1 for presence:
    dict_species_presence_vectors = {sp: np.zeros(len(df_minimal), dtype=int) for sp in species_list} 
    for i, sp in enumerate(species_list):
        dict_species_presence_vectors[sp][df_minimal.species == sp] = 1

    df_minimal = df_minimal.drop(['year', 'month', 'day', 'speciesKey', 'species', 'polygon', 'footprintWKT'], axis=1)

    for sp in species_list:
        df_minimal[sp] = dict_species_presence_vectors[sp]

    df_summary = df_minimal.groupby(['tuple_coords', 'eventDate']).sum().reset_index()
    print(f'Number of location/date combis {len(df_summary)}')
    df_summary.head()

    ## Filter by date:
    # n_original_loc_times = len(df_summary)
    # df_summary = df_summary[df_summary.eventDate >= '2016-01-01']
    # print(f'Kept {len(df_summary)}/{n_original_loc_times} time/loc combis after 2016')
    
    return df_summary, species_list

def create_species_abundance_per_loc(df_summary, species_list):
    df_summary['n_visits'] = 1  
    df_per_loc = df_summary.groupby('tuple_coords').sum().reset_index()
    df_per_loc = df_per_loc.drop(['eventDate'], axis=1)

    ## normalise by number of visits (per location)
    df_per_loc_norm = df_per_loc.copy()
    for sp in species_list:
        df_per_loc_norm[sp] = df_per_loc_norm[sp] / df_per_loc_norm['n_visits']
    return df_per_loc, df_per_loc_norm    

def create_species_presence_per_loc(df_summary, species_list):
    df_presence = df_summary.copy()
    df_presence[species_list] = df_presence[species_list].applymap(lambda x: 1 if x > 0 else 0)  # presence/absence
    if 'n_visits' not in df_presence.columns:
        df_presence['n_visits'] = 1
    df_presence = df_presence.groupby('tuple_coords').sum().reset_index()
    df_presence = df_presence.drop(columns=['eventDate'])

    for kk in species_list:
        df_presence[kk] = df_presence[kk].astype(float)  # to avoid dtype warning in next step
    for ii in range(len(df_presence)):  # normalise by number of visits
        df_presence.loc[ii, species_list] = df_presence.loc[ii, species_list] / df_presence.loc[ii, 'n_visits']
    return df_presence

def create_species_dataset(df, df_mapping_locs=None,
                           dataset_type='presence',
                           year_min=None, year_max=None,
                           threshold_n_obs_per_location=200, verbose=1,
                           folder_save=None, save_csv=False):
    if verbose:
        print('-- Starting to create species dataset. Copying.')
    df = df.copy()
    if verbose:
        print(f'-- Creating new data set from {len(df)} records.\n-- Getting locations.')
    if df_mapping_locs is None:
        df_mapping_locs, df = load_names_unique_locs(df)
    if verbose:
        print(f'-- Loaded {len(df_mapping_locs)} unique locations.\n-- Filtering by year.')
    if year_min is not None:
        assert type(year_min) == int, type(year_min)
        df = df[df.year >= year_min]
    if year_max is not None:
        assert type(year_max) == int, type(year_max)
        if year_min is not None:
            assert year_max >= year_min, (year_max, year_min)
        df = df[df.year <= year_max] 
    assert len(df) > 0, f'No records left after filtering by year {year_min} - {year_max}'
    if verbose:
        print(f'-- Kept {len(df)} records after filtering by year.\n-- Cleaning data.')
    df_clean = clean_bms_data(df, threshold_n_obs_per_location=threshold_n_obs_per_location, verbose=verbose)
    df_minimal = create_minimal_df_bms(df_clean)
    if verbose:
        print(f'-- Aggregating data to {dataset_type}.')
    df_summary, species_list = create_species_abundance_per_loc_and_date(df_minimal, verbose=verbose)
    if dataset_type == 'presence':
        df_save = create_species_presence_per_loc(df_summary, species_list)
    elif dataset_type == 'abundance':
        df_per_loc, df_per_loc_norm = create_species_abundance_per_loc(df_summary, species_list)
        df_save = df_per_loc
    
    if verbose:
        print(f'-- Matching locations coordinates to names.')
    ## match tuple coords to name_loc. Because of float errors, use isclose to match coordinates:
    dist_mat = np.zeros((len(df_save), len(df_mapping_locs)))
    for i, tc1 in enumerate(df_save.tuple_coords):
        for j, tc2 in enumerate(df_mapping_locs.tuple_coords):
            dist_mat[i, j] = np.linalg.norm(np.array(tc1) - np.array(tc2))
    idx_min = np.argmin(dist_mat, axis=1)
    assert len(idx_min) == len(df_save) and len(idx_min) == len(np.unique(idx_min)), (len(idx_min), len(df_save), len(np.unique(idx_min)))
    df_save['name_loc'] = df_mapping_locs.iloc[idx_min].name_loc.values

    ## assert that all locations match the other way around too & isclose because of float errors:
    for i_row in range(len(df_save)):
        tmp = df_mapping_locs[df_mapping_locs.name_loc == df_save.iloc[i_row].name_loc]
        assert len(tmp) == 1
        if tmp.iloc[0].tuple_coords != df_save.iloc[i_row].tuple_coords:
            assert np.isclose(tmp.iloc[0].tuple_coords[0], df_save.iloc[i_row].tuple_coords[0])

    if verbose:
        print(f'-- Created data set with {len(df_save)} locations and {len(species_list)} species.')
    if save_csv:
        print('-- Saving data.')
        if folder_save is None:
            folder_save = os.path.join(path_dict_pecl['data_folder'], f'bms_{dataset_type}')
            if not os.path.exists(folder_save):
                os.makedirs(folder_save)
        filename = f'bms_{dataset_type}_y-{year_min}-{year_max}_th-{threshold_n_obs_per_location}.csv'
        path_save = os.path.join(folder_save, filename)
        df_save.to_csv(path_save)
    if verbose:
        print('-- Done.')
    return df_save, df_summary, species_list

def load_species_dataset(folder_save=None, year_min=2018, year_max=2019, 
                         threshold_n_obs_per_location=200,
                         dataset_type='presence'):
    filename = f'bms_{dataset_type}_y-{year_min}-{year_max}_th-{threshold_n_obs_per_location}.csv'
    if folder_save is None:
        folder_save = os.path.join(path_dict_pecl['data_folder'], f'bms_{dataset_type}')
    path_save = os.path.join(folder_save, filename)
    df_save = pd.read_csv(path_save, index_col=0)
    return df_save

def plot_obs_per_year(df):
    df_tmp = df.groupby('year').size().reset_index()
    plt.scatter(df_tmp['year'], df_tmp[0])
    plt.xlabel('Year')
    plt.ylabel('Number of records')

def plot_n_obs_per_visit(df_summary, species_list, verbose=1,
                         plot_type='boxplot'):
    arr_n_species_present = np.sum(df_summary[species_list].values > 0, axis=1)
    arr_total_individuals = np.sum(df_summary[species_list].values, axis=1)

    df_tmp = {'n_species_present': arr_n_species_present, 'total_individuals': arr_total_individuals}
    df_tmp = pd.DataFrame(df_tmp)

    if plot_type == 'jointplot':
        sns.jointplot(x='n_species_present', y='total_individuals', data=df_tmp, 
                    kind='hex', joint_kws={'gridsize': 30, 'mincnt': 1}, 
                    marginal_kws={'bins': 50})
    elif plot_type == 'boxplot':
        sns.boxplot(x='n_species_present', y='total_individuals', 
                    data=df_tmp)

    plt.xlabel('Number of species present')
    plt.ylabel('Total number of individuals')

def opt_leaf(w_mat, dim=0, link_metric='correlation'):
    '''
    From popoff 
    
    create optimal leaf order over dim, of matrix w_mat.
    see also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering'''
    assert w_mat.ndim == 2
    if dim == 1:  # transpose to get right dim in shape
        w_mat = w_mat.T
    dist = scipy.spatial.distance.pdist(w_mat, metric=link_metric)  # distanc ematrix
    link_mat = scipy.cluster.hierarchy.ward(dist)  # linkage matrix
    if link_metric == 'euclidean':
        opt_leaves = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
        # print('OPTIMAL LEAF SOSRTING AND EUCLIDEAN USED')
    elif link_metric == 'correlation':
        opt_leaves = scipy.cluster.hierarchy.leaves_list(link_mat)
    return opt_leaves, (link_mat, dist)

def plot_clusters_species_per_loc(df_summary, species_list, threshold_clusters=20):
    _, df_per_loc_norm = create_species_abundance_per_loc(df_summary, species_list)

    ## cluster & sort:
    vals_species = df_per_loc_norm[species_list].values
    sorting, tmp = opt_leaf(vals_species, dim=0, link_metric='euclidean')
    vals_sorted = vals_species[sorting, :]

    fig, ax = plt.subplots(1, 3, figsize=(16, 8))

    ## plot sorted species list per ploc:
    sns.heatmap(vals_sorted, cmap='viridis', cbar=True, ax=ax[0])
    ax[0].set_xlabel('Species')
    ax[0].set_ylabel('Locations')

    ## plot dendogram:
    link_mat, dist = tmp
    dend = scipy.cluster.hierarchy.dendrogram(link_mat, ax=ax[1], 
                                            orientation='right',
                                            color_threshold=0)
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Distance')
    ax[1].set_yticks([])
    for spine in ['left', 'top', 'right']:
        ax[1].spines[spine].set_visible(False)

    ## create clusters:
    clusters = scipy.cluster.hierarchy.fcluster(link_mat, threshold_clusters, criterion='distance')
    num_clusters = len(np.unique(clusters))
    print(f'Number of clusters {num_clusters}')
    assert len(clusters) == len(df_per_loc_norm)
    df_per_loc_norm['cluster'] = clusters

    gdf_per_loc = gpd.GeoDataFrame(df_per_loc_norm, geometry=[Point(xy) for xy in df_per_loc_norm.tuple_coords])
    gdf_per_loc.crs = {'init': 'epsg:4326'}

    ## plot map UK:
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    uk = world[world['name'] == 'United Kingdom']
    uk.plot(ax=ax[2], color='white', edgecolor='black')


    gdf_per_loc.plot(markersize=6, ax=ax[2], legend=True, 
                     column='cluster', cmap='magma', categorical=True)

def get_gee_image(df_mapping_locs_row, use_point=True, verbose=0, 
                  year=None, month_start_str='06', month_end_str='09'):
    assert type(df_mapping_locs_row) == pd.Series, type(df_mapping_locs_row)
    if year is None:
        year = 2020
    if use_point:
        assert False, 'not correctly implemented yet. CRS needs to be taken into account such that buffer is in meters, not degrees.'
        print('using point')
        assert 'tuple_coords' in df_mapping_locs_row.index, df_mapping_locs_row.index
        # point = ee.Geometry.Point(row.tuple_coords)
        point = shapely.geometry.Point(df_mapping_locs_row.tuple_coords)
        polygon = point.buffer(0.01, cap_style=3)  ## buffer in degrees
        xy_coords = np.array(polygon.exterior.coords.xy).T 
        aoi = ee.Geometry.Polygon(xy_coords.tolist())
    else:
        col_polygon = 'polygon'
        assert col_polygon in df_mapping_locs_row.index and 'tuple_coords' in df_mapping_locs_row.index, df_mapping_locs_row.index
        '''buffer around polygon, because (of CRS I think) the saved tif can be rotated slightly and therefore outside pixels are blank. 
        Create larger buffer and then in download_gee_image() it is loaded, indexed and re-saved to the correct size.'''
        buffer_dist = 1000
        xy_coords = np.array(df_mapping_locs_row[col_polygon].exterior.coords.xy).T 
        aoi = ee.Geometry.Polygon(xy_coords.tolist())
        aoi = aoi.buffer(buffer_dist).bounds()

    ex_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    ## also consider creating a mosaic instead: https://gis.stackexchange.com/questions/363163/filter-out-the-least-cloudy-images-in-sentinel-google-earth-engine
    ex_im_gee = ee.Image(ex_collection 
                        #   .project(crs='EPSG:27700', scale=1)
                        .filterBounds(aoi) 
                        .filterDate(ee.Date(f'{year}-{month_start_str}-01'), ee.Date(f'{year}-{month_end_str}-01')) 
                        .select(['B4', 'B3', 'B2', 'B8'])  # 10m bands, RGB and NIR
                        .sort('CLOUDY_PIXEL_PERCENTAGE')
                        .first()  # get the least cloudy image
                        .clip(aoi))
    
    im_dims = ex_im_gee.getInfo()["bands"][0]["dimensions"]
    
    if im_dims[0] < 256 or im_dims[1] < 256:
        print('WARNING: image too small, returning None')
        return None
    
    if verbose:
        print(ex_im_gee.projection().getInfo())
        # print(f'Area AOI in km2: {aoi.area().getInfo() / 1e6}')
        print(f'Pixel dimensions: {im_dims}')
        print(ex_im_gee.getInfo()['bands'][3])
    
    return ex_im_gee

def download_gee_image(df_mapping_locs_row, use_point=False, 
                        month_start_str='06', month_end_str='09',
                       year=None, path_save=None, 
                       remove_if_too_small=True, verbose=0):
    if year is None:
        year = 2020
    im_gee = get_gee_image(df_mapping_locs_row=df_mapping_locs_row, 
                           month_start_str=month_start_str, month_end_str=month_end_str,
                           use_point=use_point, verbose=verbose, year=year)
    if im_gee is None:  ## if image was too small it was discarded
        return None, None

    name_loc = df_mapping_locs_row.name_loc

    if path_save is None:
        path_save = f'/Users/t.vanderplas/data/UKBMS_sent2_ds/sent2-4band/{year}/m-{month_start_str}-{month_end_str}/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    filepath = os.path.join(path_save, f'sent2-4band_{name_loc}_y-{year}_m-{month_start_str}-{month_end_str}.tif')
    geemap.ee_export_image(
        im_gee, filename=filepath, 
        scale=10,  # 10m bands
        file_per_band=False,# crs='EPSG:32630'
    )

    ## load & save to size correctly (because of buffer): 
    im = lca.load_tiff(filepath, datatype='da')
    desired_pixel_size = 256
    
    if verbose:
        print('Original size: ', im.shape)
    if im.shape[1] < desired_pixel_size or im.shape[2] < desired_pixel_size:
        print('WARNING: image too small, returning None')
        if remove_if_too_small:
            os.remove(filepath)
        return None, None

    ## crop:
    padding_1 = (im.shape[1] - desired_pixel_size) // 2
    padding_2 = (im.shape[2] - desired_pixel_size) // 2
    im_crop = im[:, padding_1:desired_pixel_size + padding_1, padding_2:desired_pixel_size + padding_2]
    assert im_crop.shape[0] == im.shape[0] and im_crop.shape[1] == desired_pixel_size and im_crop.shape[2] == desired_pixel_size, im_crop.shape
    if verbose:
        print('New size: ', im_crop.shape)
    im_crop.rio.to_raster(filepath)

    return im_crop, filepath

def create_timestamp(include_seconds=False):
    dt = datetime.datetime.now()
    timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
    if include_seconds:
        timestamp += ':' + str(dt.second).zfill(2)
    return timestamp

