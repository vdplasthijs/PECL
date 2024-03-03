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
import matplotlib
import torch
import torch.nn.functional as F
import paired_embeddings_models as pem

from cycler import cycler
## Create list with standard colors:
import seaborn as sns
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)

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

    gdf_bms.plot(ax=ax_map, markersize=0.5, color=color_dict_stand[0])
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
                density=True, label=r"$\cos$", edgecolor=color_dict_stand[0])
    _ = ax.hist(np.power(inner_prod, 2), bins=100, histtype='step', linewidth=1.5, # edgecolor='k',
                density=True, label=r"$\cos^2$", edgecolor=color_dict_stand[1]) 
    handles = [matplotlib.lines.Line2D([], [], c=color_dict_stand[ii]) for ii in range(2)]
    ax.legend(handles=handles, labels=[r"$\cos$", r"$\cos^2$"], loc='upper left', 
              frameon=False, bbox_to_anchor=(0.05, 1.15))
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

def get_list_timestamps_from_vnums(list_vnums, path_stats='/Users/t.vanderplas/models/PECL/stats/'):
    assert type(list_vnums) in [list, np.ndarray], f'Expected list, got {type(list_vnums)}'
    list_timestamps = []
    contents_folder = os.listdir(path_stats)
    contents_folder = [x for x in contents_folder if x.endswith('.pkl')]
    for vnum in list_vnums:
        assert type(vnum) in [int, np.int64], f'Expected int, got {type(vnum)}'
        list_candidates = [x for x in contents_folder if f'vnum-{vnum}' in x]
        assert len(list_candidates) == 1, f'Expected 1 candidate, got {len(list_candidates)}'
        list_timestamps.append(list_candidates[0])
    assert len(list_timestamps) == len(list_vnums), f'Expected {len(list_vnums)} timestamps, got {len(list_timestamps)}'
    return list_timestamps

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
    metrics_use_min = ['val_bce_loss', 'val_mse_loss']

    metrics_use = metrics_use_max + metrics_use_min
    assert metric_optimise in metrics_use, f'Optimisation metric {metric_optimise} not in metrics_use'

    df = pd.DataFrame(columns=['timestamp'] + hparams_use + metrics_use)
    for i_ts, ts in enumerate(list_ts):
        tmp_stats = dict_stats[ts]
        tmp_hparams = tmp_stats['hparams']
        tmp_df_metrics = tmp_stats['df_metrics']
        print(tmp_stats['df_metrics'].columns)

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
        for h in hparams_use:
            assert h in tmp_hparams.keys(), f'Hyperparameter {h} not in tmp_hparams'
        for m in metrics_use:
            assert m in tmp_df_metrics.columns, f'Metric {m} not in tmp_df_metrics'
        tmp_row = [ts] + [tmp_hparams[h] for h in hparams_use] + [tmp_metrics[m] for m in metrics_use]
        df.loc[len(df)] = tmp_row

    return df, ('timestamp', hparams_use, metrics_use, metric_optimise)

def create_printable_table(df, hparams_use, metrics_use, metric_optimise='val_top_10_acc',
                           hparam_show=[], metrics_show=['val_top_10_acc', 'val_top_5_acc', 'val_mse_loss'],
                           col_seed='seed_used', save_table=False, filename=None,
                           folder_save=os.path.join(path_dict_pecl['repo'], 'tables/'),
                           caption_tex=None, label_tex=None, position_tex='h',
                           highlight_best_row=False):
    
    dict_rename_metrics = {'val_top_20_acc': 'Top-20',
                    'val_top_10_acc': 'Top-10',
                    'val_top_5_acc': 'Top-5',
                    'val_top_1_acc': 'Top-1',
                    'val_bce_loss': 'BCE',
                    'val_mse_loss': 'MSE',
                    'val_pecl-softmax_loss': 'PECL'}
    dict_rename_hparams = {
        'species_process': 'Species',
        'alpha_ratio_loss': r"$\alpha$",
        'batch_size_used': 'Batch',
        'fix_seed': 'Seed',
        'freeze_resnet': 'Freeze Res',
        'lr': 'Learning rate',
        'n_enc_channels': 'Channels',
        'pecl_knn': 'KNN',
        'pecl_knn_hard_labels': 'Hard labels'
        }
    unique_vals_ignore = ['time_created', 'n_epochs_converged']
    
    ## Drop hparams with only one unique value (not relevant for comparison)
    df_num_val = df.copy()  # this df will be reformatted, but maintain numeric values (while df_tex will be formatted as str for latex)
    cols_drop = []
    cols_vals_all_same = {}
    cols_multiple_values = []
    for h in hparams_use:
        if h in unique_vals_ignore:
            cols_drop.append(h)
            continue 
        n_unique = df_num_val[h].nunique()
        if n_unique == 1:
            cols_vals_all_same[h] = df_num_val[h].unique()[0]
            cols_drop.append(h)
        else:
            cols_multiple_values.append(h)
            print(f'Hyperparameter {h} has {n_unique} unique values')
    df_num_val = df_num_val.drop(columns=cols_drop)
    # print(cols_vals_all_same)
    if col_seed in cols_multiple_values:
        cols_multiple_values.remove(col_seed)
        multiple_seeds = True
    else:
        assert col_seed in cols_vals_all_same.keys(), f'Column {col_seed} not in cols_vals_all_same'
        print(f'Column {col_seed} has only one unique value: {cols_vals_all_same[col_seed]}')
        multiple_seeds = False
    
    if hparam_show == []:
        hparam_show = cols_multiple_values
        print(f'No hyperparameters to show specified, using {hparam_show}')

    ## Drop metrics not in metrics_show  
    for m in metrics_use:
        if m not in metrics_show:
            df_num_val = df_num_val.drop(columns=[m])

    df_num_val = df_num_val.rename(columns=dict_rename_metrics)
    df_num_val = df_num_val.drop(columns=['timestamp'])

    ## compute mean and sem across seeds:
    if multiple_seeds:
        assert col_seed in df_num_val.columns
        assert df_num_val[col_seed].nunique() > 1
        df_num_val = df_num_val.drop(columns=[col_seed])
    df_num_val = df_num_val.groupby(hparam_show).agg(['mean', 'sem'])  # mean and sem across seeds. With only one seed, sem is NaN

    ## Find all columns that arent hparams & rewrite to mean pm sem
    cols_metrics = []
    for m in df_num_val.columns:
        # print(m)
        if m[0] in hparams_use:
            continue
        else:
            if m[0] not in cols_metrics:  # avoid duplicates because of mean and sem
                cols_metrics.append(m[0])
    
    ## Scale and format values
    formatted_vals_dict = {}
    col_renaming_dict = {}
    for m in cols_metrics:
        if 'Top-' in m:
            scale = 100 
            new_name = m + ' [\%]'
            n_decimals = 1
        else:    
            max_val = df_num_val[m].loc[:, ('mean')].max()
            ## scale so that first digit is before decimal point
            scale = 10 ** -(int(np.log10(max_val)) - 1)
            n_decimals = 2
            if scale == 1:
                new_name = m 
            else:   
                new_name = m + f' [{1 / scale:.0e}]'
        col_renaming_dict[m] = new_name
        scaled_col = df_num_val[m] * scale
        if n_decimals == 2:
            formatted_vals_dict[new_name] = scaled_col.apply(lambda x: f'{x["mean"]:.2f} ' + r"$\pm$" + f' {x["sem"]:.2f}', axis=1)
        elif n_decimals == 1:
            formatted_vals_dict[new_name] = scaled_col.apply(lambda x: f'{x["mean"]:.1f} ' + r"$\pm$" + f' {x["sem"]:.1f}', axis=1)
        else:
            assert False, f'Unexpected number of decimals: {n_decimals}'
    df_tex = pd.DataFrame(formatted_vals_dict)
    df_tex = df_tex.reset_index()
    df_num_val = df_num_val.reset_index()

    if highlight_best_row:
        metrics_use_max = ['val_top_20_acc', 'val_top_10_acc', 'val_top_5_acc', 'val_top_1_acc']
        metrics_use_min = ['val_bce_loss', 'val_mse_loss', 'val_pecl-softmax_loss']
        metrics_use_max = [dict_rename_metrics[m] for m in metrics_use_max]
        metrics_use_min = [dict_rename_metrics[m] for m in metrics_use_min]
        for m in metrics_use_max + metrics_use_min:
            if m not in df_num_val.droplevel(1, axis=1).columns:
                continue
            if m in metrics_use_max:
                best_row = df_num_val[m]['mean'].idxmax()
            elif m in metrics_use_min:
                best_row = df_num_val[m]['mean'].idxmin()
            new_val = '\\textbf{' + df_tex[col_renaming_dict[m]].loc[best_row] + '}'
            df_tex.at[best_row, col_renaming_dict[m]] = new_val
   
    df_tex = df_tex.rename(columns=dict_rename_hparams)
    for c in df_tex.columns:
        if df_tex[c].dtype == 'float64' or df_tex[c].dtype == 'float32':
            df_tex[c] = df_tex[c].apply(lambda x: str(x))

    if save_table:
        assert filename is not None, 'Filename not specified'
        assert os.path.exists(folder_save), f'Folder {folder_save} does not exist'
        assert filename.endswith('.tex'), f'Filename {filename} does not end with .tex'
        path_save = os.path.join(folder_save, filename)
        df_tex.to_latex(path_save, index=False, escape=False, na_rep='NaN',
                caption=caption_tex, label=label_tex, position=position_tex)

    return df_num_val, df_tex

def print_table_batchsize():

    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=np.arange(199, 217)
    ))

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different batch sizes. ' \
              'The best performing model for each metric is highlighted in bold.'

    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                           metric_optimise=tmp_details[3], # hparam_show='alpha_ratio_loss',
                           save_table=True, filename='tab_batch-size.tex', highlight_best_row=True,
                           label_tex='tab:batch_size', caption_tex=caption)
    
    return (df_num_val, df_tex)

def print_table_alpha():
    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=np.arange(164, 171)
    ))

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different values of the $\alpha$ ratio loss hyperparameter. ' \
              'The best performing model for each metric is highlighted in bold.'
    
    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                            metric_optimise=tmp_details[3], # hparam_show='alpha_ratio_loss',
                            save_table=True, filename='tab_alpha-ratio.tex', highlight_best_row=True,
                            label_tex='tab:alpha_ratio', caption_tex=caption)
    return (df_num_val, df_tex)