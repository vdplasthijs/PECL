import os, sys, copy
import ast
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

sys.path.append('/Users/t.vanderplas/repos/reproducible_figures/scripts/')
import rep_fig_vis as rfv
rfv.set_fontsize(10)
# sys.path.append(os.path.join(path_dict_pecl['repo'], 'content/'))
# import api_keys
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()
# import create_dataset_utils as cdu 

fig_folder = '/Users/t.vanderplas/repos/PECL/figures/'

def plot_stats_df_presence(ds, ax_hist_visits=None, ax_hist_species=None,
                           ax_hist_species_log=None, ax_map=None, path_map=None,
                           plot_type_species_count='line'):
    if ax_hist_visits is None or ax_hist_species is None or ax_hist_species_log is None:
        # fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        fig, ax = plt.subplots(1, 4, figsize=(10, 2))

        ax_hist_visits = ax.flatten()[0]
        ax_hist_species = ax.flatten()[1]
        ax_hist_species_log = ax.flatten()[2]
        ax_map = ax.flatten()[3]

    _ = ax_hist_visits.hist(ds.df_presence.n_visits, 
                bins=np.linspace(0, ds.df_presence.n_visits.max() + 1, 20), 
                edgecolor='k')
    ax_hist_visits.set_xlabel('Visits per location')
    ax_hist_visits.set_ylabel('Frequency')

    species_count = ds.df_presence[ds.species_list].mean()
    species_count = np.sort(species_count)[::-1]

    if plot_type_species_count == 'line':
        _ = ax_hist_species.plot(np.arange(len(species_count)), 
                                species_count, '.-')
        _ = ax_hist_species_log.plot(np.arange(len(species_count)), 
                                    species_count, '.-')

    elif plot_type_species_count == 'bar':
        _ = ax_hist_species.bar(x=np.arange(len(species_count)), 
                                height=species_count, edgecolor='k')
        _ = ax_hist_species_log.bar(x=np.arange(len(species_count)), 
                                    height=species_count, edgecolor='k')

    ax_hist_species_log.set_yscale('log')
    for ax_ in [ax_hist_species, ax_hist_species_log]:    
        ax_.set_xlabel('Sorted species ID')
        ax_.set_ylabel('P(presence)')

    for ax_ in [ax_hist_species, ax_hist_species_log, ax_hist_visits]:
        rfv.despine(ax_)

    if path_map is None:
        ## https://www.diva-gis.org/gdata
        path_map = '/Users/t.vanderplas/data/GBR_adm/GBR_adm0.shp'
    gdf_uk = gpd.read_file(path_map)
    gdf_uk.plot(ax=ax_map, color='k', alpha=0.5)
    # print(gdf_uk.crs)
    gdf_uk.crs = 'epsg:27700'
    # print(gdf_uk.crs)

    point_locs = ds.df_presence.tuple_coords
    # return point_locs
    point_locs = [shapely.geometry.Point(ast.literal_eval(loc)) for loc in point_locs]
    gdf_bms = gpd.GeoDataFrame(geometry=point_locs)

    gdf_bms.plot(ax=ax_map, color='r', markersize=1)
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-8, 2)
    ax_map.axis('off')
    ax_map.set_title('UKBMS locations')

def dataset_fig(ds, save_fig=False,
                example_inds=[86, 190, 343, 777, 898, 1000]):
    # fig, ax = plt.subplots(2, 4, figsize=(10, 5), 
    #                        gridspec_kw={'wspace': 0.5, 'hspace': 0.5})  
    n_examples = len(example_inds)
    fig = plt.figure(figsize=(10, 5))
    gs_top = fig.add_gridspec(1, 4, wspace=0.5, hspace=0.5, top=0.95, bottom=0.55, left=0.05, right=0.98)
    gs_bottom = fig.add_gridspec(1, n_examples, wspace=0.1, hspace=0.5, top=0.45, bottom=0.02, left=0.02, right=0.97)
    ax_top = [fig.add_subplot(gs_top[i]) for i in range(4)]
    ax_bottom = [fig.add_subplot(gs_bottom[i]) for i in range(n_examples)]

    plot_stats_df_presence(ds, ax_hist_visits=ax_top[0], ax_hist_species=ax_top[1],
                            ax_hist_species_log=ax_top[2], ax_map=ax_top[3])
    
    
    for i, ind in enumerate(example_inds):
        ax_ = ax_bottom[i]
        ax_, species_ax = ds.plot_image(ind, ax=ax_)   
        ax_.set_title(f'Example {ind}')
        ax_.axis('off')

        # if i == 0:
            # ax_.set_ylabel('Example images')
        # if i == n_examples - 1:
        #     species_ax.set_ylabel('Species ID')

    ax_.annotate('P(presence)', xy=(1.15, 0.5), xycoords='axes fraction', 
                 va='center', ha='center',
                 rotation=90)

    plt.draw()
    rfv.add_panel_label(ax_top[0], label_letter='a', fontsize=14)
    rfv.add_panel_label(ax_top[1], label_letter='b', fontsize=14)
    rfv.add_panel_label(ax_top[3], label_letter='c', fontsize=14)
    rfv.add_panel_label(ax_bottom[0], label_letter='d', fontsize=14, x_offset=0.2)

    if save_fig:
        plt.savefig(os.path.join(fig_folder, 'dataset_overview.pdf'), dpi=300, bbox_inches='tight')