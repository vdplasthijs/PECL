## Author: Thijs van der Plas

import os, sys 
import numpy as np 
import pandas as pd 
from dwca.read import DwCAReader
import matplotlib.pyplot as plt
import seaborn as sns 
import shapely 
from shapely.geometry import Point, Polygon
# import warnings
# from shapely.errors import ShapelyDeprecationWarning
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
import geopandas as gpd
# import h3pandas 
sys.path.append('../reproducible_figures/scripts/')
import rep_fig_vis as rfv
from tqdm import tqdm
import loadpaths_pecl
import scipy 
from matplotlib.colors import ListedColormap
import scipy.spatial
import scipy.cluster

sys.path.append('../../../cnn-land-cover/scripts/')
import land_cover_analysis as lca 
import land_cover_visualisation as lcv
import land_cover_models as lcm 

path_dict_pecl = loadpaths_pecl.loadpaths()

def load_df_gbif(path_gbif_ds=None, verbose=1):
    if path_gbif_ds is None:
        path_gbif_ds = path_dict_pecl['ukbms_full_dataset'] 
    with DwCAReader(path_gbif_ds) as dwca:
        df = dwca.pd_read('occurrence.txt', parse_dates=True)

    # df = df[df['institutionCode'] == 'UKBMS']
        
    if verbose:
        print('Loaded GBIF dataset with {} records'.format(len(df)))
    return df

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

def create_species_presence_per_loc_and_date(df_minimal, verbose=1):
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

def create_species_presence_per_loc(df_summary, species_list):
    df_summary['n_visits'] = 1  
    df_per_loc = df_summary.groupby('tuple_coords').sum().reset_index()
    df_per_loc = df_per_loc.drop(['eventDate'], axis=1)

    ## normalise by number of visits (per location)
    df_per_loc_norm = df_per_loc.copy()
    for sp in species_list:
        df_per_loc_norm[sp] = df_per_loc_norm[sp] / df_per_loc_norm['n_visits']
    return df_per_loc, df_per_loc_norm    

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
    _, df_per_loc_norm = create_species_presence_per_loc(df_summary, species_list)

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
