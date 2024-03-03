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
import torch
import torch.nn.functional as F
import paired_embeddings_models as pem

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

    gdf_bms.plot(ax=ax_map, color='r', markersize=0.5)
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-8.2, 2)
    ax_map.set_ylim(49, 61)
    ax_map.axis('off')
    ax_map.legend(['S2-BMS location'], loc='lower left', fontsize=8, bbox_to_anchor=(0, -.25))
    # ax_map.set_title('UKBMS locations')

def dataset_fig(ds, all_labels=None, save_fig=False,
                example_inds=[86, 190, 343, 777, 898, 1000]):
    
    n_examples = len(example_inds)
    n_plots_top = 5
    fig = plt.figure(figsize=(10, 4))
    gs_top = fig.add_gridspec(1, n_plots_top, wspace=0.5, hspace=0.5, top=0.95, bottom=0.6, left=0.02, right=0.98)
    gs_bottom = fig.add_gridspec(1, n_examples, wspace=0.1, hspace=0.5, top=0.45, bottom=0.02, left=0.02, right=0.97)
    ax_top = [fig.add_subplot(gs_top[i]) for i in range(n_plots_top)]
    ax_bottom = [fig.add_subplot(gs_bottom[i]) for i in range(n_examples)]

    plot_stats_df_presence(ds, ax_hist_visits=ax_top[1], ax_hist_species=ax_top[2],
                            ax_hist_species_log=ax_top[3], ax_map=ax_top[0])
    if all_labels is not None:
        ax_ = ax_top[4]
        plot_distr_label_inner_prod(all_labels, ax=ax_)

    
    for i, ind in enumerate(example_inds):
        ax_ = ax_bottom[i]
        ax_, species_ax = ds.plot_image(ind, ax=ax_)   
        ax_.set_title(f'Example {ind}')
        ax_.axis('off')

    ax_.annotate('P(presence)', xy=(1.15, 0.5), xycoords='axes fraction', 
                 va='center', ha='center',
                 rotation=90)

    plt.draw()
    rfv.add_panel_label(ax_top[0], label_letter='a', fontsize=14, x_offset=0.2)
    rfv.add_panel_label(ax_top[1], label_letter='b', fontsize=14)
    rfv.add_panel_label(ax_top[2], label_letter='c', fontsize=14)
    rfv.add_panel_label(ax_top[4], label_letter='d', fontsize=14)
    rfv.add_panel_label(ax_bottom[0], label_letter='e', fontsize=14, x_offset=0.2)

    if save_fig:
        plt.savefig(os.path.join(fig_folder, 'dataset_overview.pdf'), dpi=300, bbox_inches='tight')

def plot_distr_label_inner_prod(all_labels, ax=None, save_fig=False):
    assert type(all_labels) == np.ndarray, f'Expected numpy array, got {type(all_labels)}'
    n_species = all_labels.shape[1]
    n_labels = all_labels.shape[0]
    print(f'Number of species: {n_species}, number of labels: {n_labels}')    

    inner_prod = np.dot(all_labels, all_labels.T)
    assert inner_prod.shape == (n_labels, n_labels)
    triu_inds = np.triu_indices(n_labels, k=1)
    inner_prod = inner_prod[triu_inds]
    assert inner_prod.shape == (n_labels * (n_labels - 1) // 2, )
    assert inner_prod.ndim == 1
    print(f'Inner product shape: {inner_prod.shape}')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    _ = ax.hist(inner_prod, bins=100, histtype='step', linewidth=1.5, # edgecolor='k',
                density=True, label='cos sim')
    
    _ = ax.hist(np.power(inner_prod, 2), bins=100, histtype='step', linewidth=1.5, # edgecolor='k',
                density=True, label='pow cos sim')
    ax.legend()
    ax.set_xlabel('cos similarity ' + r'$s_{ij}$')
    # ax.set_ylabel('Number of pairs')
    ax.set_ylabel('Density of pairs')
    rfv.despine(ax)

    if save_fig:
        plt.savefig(os.path.join(fig_folder, 'distr_inner_prod_labels.pdf'),
                                 bbox_inches='tight')

    return ax

def stack_all_labels(ds, normalise=True):
    all_labels = []
    for sample in tqdm(ds):
        all_labels.append(sample[1][None, :])
    all_labels = torch.cat(all_labels, dim=0)
    if normalise:
        all_labels_norm = F.normalize(all_labels, p=2, dim=1)
        all_labels_norm = all_labels_norm.detach().cpu().numpy()
    else:
        all_labels_norm = None
    all_labels = all_labels.detach().cpu().numpy()
    return all_labels, all_labels_norm

def load_list_timestamps(list_ts):
    assert type(list_ts) == list, f'Expected list, got {type(list_ts)}'
    dict_stats = {}
    for ts in list_ts:
        tmp_stats = pem.load_stats(timestamp=ts)
        dict_stats[ts] = tmp_stats
    return dict_stats

def create_df_list_timestamps(list_ts, metric_optimise='val_top_10_acc'):
    dict_stats = load_list_timestamps(list_ts)
    example_stats = dict_stats[list_ts[0]]
    hparams_exclude = ['class_weights']
    hparams_use = [h for h in  example_stats['hparams'].keys() if h not in hparams_exclude]

    metrics_use_max = ['val_top_20_acc', 'val_top_10_acc', 'val_top_5_acc', 'val_top_1_acc']
    metrics_use_min = ['val_bce_loss', 'val_mse_loss', 'val_pecl-softmax_loss']

    metrics_use = metrics_use_max + metrics_use_min
    assert metric_optimise in metrics_use, f'Optimisation metric {metric_optimise} not in metrics_use'

    df = pd.DataFrame(columns=['timestamp'] + hparams_use + metrics_use)
    for i_ts, ts in enumerate(list_ts):
        tmp_stats = dict_stats[ts]
        tmp_hparams = tmp_stats['hparams']
        tmp_df_metrics = tmp_stats['df_metrics']

        if i_ts == 0:
            hparams_previous = tmp_hparams.keys()
            metrics_previous = tmp_df_metrics.columns
        else:
            assert hparams_previous == tmp_hparams.keys(), 'Hyperparameters not consistent'
            assert np.all(metrics_previous == tmp_df_metrics.columns), 'Metrics not consistent'

        if metric_optimise in metrics_use_max:
            ind_epoch_best = tmp_df_metrics[metric_optimise].idxmax()
        else:
            ind_epoch_best = tmp_df_metrics[metric_optimise].idxmin()

        tmp_metrics = tmp_df_metrics.loc[ind_epoch_best]
        tmp_row = [ts] + [tmp_hparams[h] for h in hparams_use] + [tmp_metrics[m] for m in metrics_use]
        df.loc[len(df)] = tmp_row

    return df, ('timestamp', hparams_use, metrics_use, metric_optimise)

def create_printable_table(df, hparams_use, metrics_use, metric_optimise='val_top_10_acc',
                           hparam_show=[], metrics_show=['val_top_10_acc', 'val_top_5_acc', 'val_mse_loss'],
                           col_seed='seed_used'):
    df_print = df.copy()
    unique_vals_ignore = ['time_created', 'n_epochs_converged']
    cols_drop = []
    vals_all_same = {}
    for h in hparams_use:
        if h in unique_vals_ignore:
            cols_drop.append(h)
            continue 
        n_unique = df_print[h].nunique()
        if n_unique == 1:
            vals_all_same[h] = df_print[h].unique()[0]
            cols_drop.append(h)
        else:
            print(f'Hyperparameter {h} has {n_unique} unique values')
    df_print = df_print.drop(columns=cols_drop)
    print(vals_all_same)
    # return df_print
    
    for m in metrics_use:
        if m not in metrics_show:
            df_print = df_print.drop(columns=[m])

    dict_rename = {'val_top_20_acc': 'Top 20 acc',
                    'val_top_10_acc': 'Top 10 acc',
                    'val_top_5_acc': 'Top 5 acc',
                    'val_top_1_acc': 'Top 1 acc',
                    'val_bce_loss': 'BCE loss',
                    'val_mse_loss': 'MSE loss',
                    'val_pecl-softmax_loss': 'PECL loss'}
    dict_rename = {k: v for k, v in dict_rename.items() if k in df_print.columns}
    df_print = df_print.rename(columns=dict_rename)
    df_print = df_print.drop(columns=['timestamp'])

    ## compute mean and sem across seeds:
    assert col_seed in df_print.columns
    assert df_print[col_seed].nunique() > 1
    df_print = df_print.drop(columns=[col_seed])
    df_print = df_print.groupby(hparam_show).agg(['mean', 'sem'])#.reset_index()

    ## Find all columns that arent hparams & rewrite to mean pm sem
    cols_metrics = []
    for m in df_print.columns:
        print(m)
        if m[0] in hparam_show:
            continue
        else:
            if m[0] not in cols_metrics:
                cols_metrics.append(m[0])
    
    tmp_dict = {}
    for m in cols_metrics:
        tmp_dict[m] = df_print[m].apply(lambda x: f'{x["mean"]:.3f} ' +r"$\pm$" + f' {x["sem"]:.3f}', axis=1)

    new_df = pd.DataFrame(tmp_dict)
    new_df = new_df.reset_index()
    df_print = df_print.round(3)

    return df_print, new_df