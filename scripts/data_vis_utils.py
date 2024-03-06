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

def plot_data_split_stats(path_split=os.path.join(path_dict_pecl['repo'], 'content/split_indices_2024-03-04-1831.pth')):
    split_indices = torch.load(path_split)
    train_inds = split_indices['train_indices']
    val_inds = split_indices['val_indices']
    test_inds = split_indices['test_indices']
    clusters = split_indices['clusters']

    clusters_train = clusters[train_inds]
    clusters_val = clusters[val_inds]
    clusters_test = clusters[test_inds]

    assert len(np.intersect1d(train_inds, val_inds)) == 0
    assert len(np.intersect1d(train_inds, test_inds)) == 0
    assert len(np.intersect1d(val_inds, test_inds)) == 0
    assert len(np.intersect1d(clusters_train, clusters_val)) == 0
    assert len(np.intersect1d(clusters_train, clusters_test)) == 0
    assert len(np.intersect1d(clusters_val, clusters_test)) == 0

    dataset_split = np.zeros(len(clusters))
    dataset_split[train_inds] = 1
    dataset_split[val_inds] = 2
    dataset_split[test_inds] = 3

    print(f'Number of training samples: {len(train_inds)}, fraction: {len(train_inds) / len(clusters):.2f}')
    print(f'Number of validation samples: {len(val_inds)}, fraction: {len(val_inds) / len(clusters):.2f}')
    print(f'Number of test samples: {len(test_inds)},   fraction: {len(test_inds) / len(clusters):.2f}')
    return dataset_split, clusters

def dataset_fig(ds, all_labels=None, save_fig=False,
                title_examples=True,
                example_inds=[126, 1000, 167, 370, 457, 635]):
    example_inds = np.sort(example_inds)
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
        _ = plot_distr_label_inner_prod(all_labels, ax=ax_)

    
    for i, ind in enumerate(example_inds):
        ax_ = ax_bottom[i]
        ax_, species_ax = ds.plot_image(ind, ax=ax_)   
        if title_examples:
            ax_.set_title(f'Example #{ind}')
        else:
            ax_.set_title('')
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

    return ax, inner_prod
    

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

def get_mean_rates_results(use_precomputed=True, train_test_filepath='../../content/split_indices_2024-03-04-1831.pth'):
    '''This takes a minute to compute, so have copy-pasted results for quick access.'''
    
    if use_precomputed:  # uses '../../content/split_indices_2024-03-04-1831.pth'
        test_loss_dict = {'mae': [0.0647730901837349],
                        'mse': [0.013860139064490795],
                        'bce': [0.2443312406539917],
                        'top_5': [0.5870967507362366],
                        'top_10': [0.6731182336807251],
                        'top_20': [0.8432795405387878]}
    else:
        ds = pem.DataSetImagePresence(image_folder='/Users/t.vanderplas/data/UKBMS_sent2_ds/sent2-4band/mix-2018-2019/m-06-09/',
                              presence_csv='/Users/t.vanderplas/data/UKBMS_sent2_ds/bms_presence/bms_presence_y-2018-2019_th-200.csv',
                              species_process='all',
                              zscore_im=True, 
                              augment_image=True, mode='val')

        train_ds, val_ds, test_ds = ds.split_into_train_val(filepath=train_test_filepath)
        assert test_ds is not None, 'Test set not found'
        mean_rates = pem.MeanRates(train_ds=train_ds, val_ds=test_ds)
        test_loss_dict = mean_rates.val_loss_dict
    return test_loss_dict


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

def create_df_list_timestamps(list_ts, split_use='test'):
    dict_stats = load_list_timestamps(list_ts)
    example_stats = dict_stats[list_ts[0]]
    hparams_exclude = ['class_weights']
    hparams_use = [h for h in  example_stats['hparams'].keys() if h not in hparams_exclude]

    if split_use == 'val':
        metric_optimise='val_top_10_acc'
        metrics_use_max = ['val_top_20_acc', 'val_top_10_acc', 'val_top_5_acc', 'val_top_1_acc']
        metrics_use_min = ['val_bce_loss', 'val_mse_loss']
        metrics_use = metrics_use_max + metrics_use_min
        assert metric_optimise in metrics_use, f'Optimisation metric {metric_optimise} not in metrics_use'
        col_df_use = 'df_metrics'
    elif split_use == 'test':
        metric_optimise = None
        metrics_use_max = ['test_top_20_acc', 'test_top_10_acc', 'test_top_5_acc', 'test_top_1_acc']
        metrics_use_min = ['test_bce_loss', 'test_mse_loss', 'test_mae_loss']
        metrics_use = metrics_use_max + metrics_use_min
        col_df_use = 'test_metrics'

    df = pd.DataFrame(columns=['timestamp'] + hparams_use + metrics_use)
    for i_ts, ts in enumerate(list_ts):
        tmp_stats = dict_stats[ts]
        tmp_hparams = tmp_stats['hparams']
        tmp_df_metrics = tmp_stats[col_df_use]
        
        if i_ts == 0:
            hparams_previous = tmp_hparams.keys()
            metrics_previous = tmp_df_metrics[metrics_use].columns
        else:
            assert hparams_previous == tmp_hparams.keys(), 'Hyperparameters not consistent'
            assert np.all(metrics_previous == tmp_df_metrics[metrics_use].columns), 'Metrics not consistent'

        if split_use == 'val':
            if metric_optimise in metrics_use_max:
                ind_epoch_best = tmp_df_metrics[metric_optimise].idxmax()
            else:
                ind_epoch_best = tmp_df_metrics[metric_optimise].idxmin()
        elif split_use == 'test':
            ind_epoch_best = tmp_df_metrics.index[-1]
            assert ind_epoch_best == 0, f'Expected last epoch, got {ind_epoch_best}'
            
        tmp_metrics = tmp_df_metrics.loc[ind_epoch_best]
        for h in hparams_use:
            assert h in tmp_hparams.keys(), f'Hyperparameter {h} not in tmp_hparams'
        for m in metrics_use:
            assert m in tmp_df_metrics.columns, f'Metric {m} not in tmp_df_metrics'
        tmp_row = [ts] + [tmp_hparams[h] for h in hparams_use] + [tmp_metrics[m] for m in metrics_use]
        df.loc[len(df)] = tmp_row

    return df, ('timestamp', hparams_use, metrics_use, metric_optimise)

def create_df_val_timeseries(list_ts, n_epochs_expected=51):
    dict_stats = load_list_timestamps(list_ts)
    example_stats = dict_stats[list_ts[0]]
    hparams_exclude = ['class_weights']
    hparams_use = [h for h in  example_stats['hparams'].keys() if h not in hparams_exclude]
    metrics_use_max = ['val_top_20_acc', 'val_top_10_acc', 'val_top_5_acc', 'val_top_1_acc']
    metrics_use_min = ['val_bce_loss', 'val_mse_loss', 'val_mae_loss']
    metrics_use = metrics_use_max + metrics_use_min
    col_df_use = 'df_metrics'

    dict_metrics = {}
    for i_ts, ts in enumerate(list_ts):
        ts_name = ts.split('_')[1]
        tmp_stats = dict_stats[ts]
        tmp_hparams = tmp_stats['hparams']
        tmp_hparams = {k: tmp_hparams[k] for k in hparams_use}
        tmp_hparams['timestamp'] = ts_name
        tmp_df_metrics = tmp_stats[col_df_use]
        dict_metrics[ts_name] = tmp_df_metrics[metrics_use]
        assert len(tmp_df_metrics) == n_epochs_expected, f'Expected {n_epochs_expected} epochs, got {len(tmp_df_metrics)}'

        if i_ts == 0:
            hparams_previous = tmp_hparams.keys()
            metrics_previous = tmp_df_metrics[metrics_use].columns
        else:
            assert hparams_previous == tmp_hparams.keys(), 'Hyperparameters not consistent'
            assert np.all(metrics_previous == tmp_df_metrics[metrics_use].columns), 'Metrics not consistent'
        
        if i_ts == 0:
            df_hparams = pd.DataFrame(tmp_hparams, index=[0])
        else:
            df_hparams = pd.concat([df_hparams, pd.DataFrame(tmp_hparams, index=[0])], axis=0).reset_index(drop=True)

    assert len(df_hparams) == len(list_ts), f'Expected {len(list_ts)} rows, got {len(df_hparams)}'
    assert len(df_hparams) == df_hparams['timestamp'].nunique(), 'Timestamps not unique'

    list_unique_cols = []
    list_ident_cols = []
    for c in df_hparams.columns:
        if df_hparams[c].nunique() == 1:
            list_ident_cols.append(c)
        else:
            list_unique_cols.append(c)

    df_hparams_unique = df_hparams[list_unique_cols]
    df_hparams_ident = df_hparams[list_ident_cols]

    return (df_hparams_unique, df_hparams_ident, dict_metrics)

def create_printable_table(df, hparams_use, metrics_use, split_use='test', 
                           hparam_show=[], add_mean_rates=False,
                           col_seed='seed_used', save_table=False, filename=None,
                           folder_save=os.path.join(path_dict_pecl['repo'], 'tables/'),
                           caption_tex=None, label_tex=None, position_tex='h',
                           highlight_best_row=False, drop_columns_tex=[]):
    if split_use == 'val':
        metrics_show = ['val_top_10_acc', 'val_top_5_acc', 'val_mse_loss']
        dict_rename_metrics = {'val_top_20_acc': 'Top-20',
                        'val_top_10_acc': 'Top-10',
                        'val_top_5_acc': 'Top-5',
                        'val_top_1_acc': 'Top-1',
                        'val_bce_loss': 'BCE',
                        'val_mse_loss': 'MSE',
                        'val_pecl-softmax_loss': 'PECL'}
    elif split_use == 'test':
        metrics_show = ['test_top_10_acc', 'test_top_5_acc', 'test_mse_loss']
        dict_rename_metrics = {'test_top_20_acc': 'Top-20',
                        'test_top_10_acc': 'Top-10',
                        'test_top_5_acc': 'Top-5',
                        'test_top_1_acc': 'Top-1',
                        'test_bce_loss': 'BCE',
                        'test_mse_loss': 'MSE',
                        'test_mae_loss': 'MAE',
                        'test_pecl-softmax_loss': 'PECL'}
    dict_rename_hparams = {
        'species_process': 'Species',
        'alpha_ratio_loss': r"$\alpha$",
        'batch_size_used': 'Batch',
        'fix_seed': 'Seed',
        'freeze_resnet': 'Freeze Res',
        'lr': 'LR',
        'n_enc_channels': 'Channels',
        'pecl_knn': '$k$',
        'pecl_knn_hard_labels': 'Hard labels',
        'p_dropout': 'p(dropout)',
        'n_layers_mlp_pred': 'MLP',
        'pretrained_resnet': 'Model'
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
    # print(df_num_val)
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
            print(f'Dropping metric {m}')
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
        if m[0] in hparams_use:
            continue
        else:
            if m[0] not in cols_metrics:  # avoid duplicates because of mean and sem
                cols_metrics.append(m[0])
    
    if add_mean_rates:
        mr_test_loss_dict = get_mean_rates_results(use_precomputed=True)
        df_mean_rates = pd.DataFrame(mr_test_loss_dict)
        dict_rename_metrics_mr = {'mae': 'MAE', 'mse': 'MSE', 'bce': 'BCE', 'top_5': 'Top-5', 'top_10': 'Top-10', 'top_20': 'Top-20'}
        df_mean_rates = df_mean_rates.rename(columns=dict_rename_metrics_mr)
        for m in df_mean_rates.columns:
            if m not in [dict_rename_metrics[x] for x in metrics_show]:
                df_mean_rates = df_mean_rates.drop(columns=[m])
                print(f'Dropping metric {m} from mean rates')
        
    ## Scale and format values
    formatted_vals_dict = {}
    col_renaming_dict = {}
    scale_dict = {}
    decimals_dict = {}
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
        decimals_dict[m] = n_decimals
        scale_dict[m] = scale
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
        if split_use == 'val':
            metrics_use_max = ['val_top_20_acc', 'val_top_10_acc', 'val_top_5_acc', 'val_top_1_acc']
            metrics_use_min = ['val_bce_loss', 'val_mse_loss', 'val_pecl-softmax_loss']
        elif split_use == 'test':
            metrics_use_max = ['test_top_20_acc', 'test_top_10_acc', 'test_top_5_acc', 'test_top_1_acc']
            metrics_use_min = ['test_bce_loss', 'test_mse_loss', 'test_mae_loss', 'test_pecl-softmax_loss']
        metrics_use_max = [dict_rename_metrics[m] for m in metrics_use_max]
        metrics_use_min = [dict_rename_metrics[m] for m in metrics_use_min]
        for m in metrics_use_max + metrics_use_min:
            if m not in df_num_val.droplevel(1, axis=1).columns:
                continue
            if m in metrics_use_max:
                best_row = df_num_val[m]['mean'].idxmax()
            elif m in metrics_use_min:
                best_row = df_num_val[m]['mean'].idxmin()
            best_val = df_num_val[m]['mean'].loc[best_row]
            ## assert better than mean_rates
            if add_mean_rates:
                if m in df_mean_rates.columns:
                    # print(m, best_val, df_mean_rates[m].max(), df_mean_rates[m].min())
                    if m in metrics_use_max:
                        assert df_mean_rates[m].max() <= best_val, f'Best row not better than mean rates for {m}, {df_mean_rates[m].max()} vs {best_val}'
                    elif m in metrics_use_min:
                        assert df_mean_rates[m].min() >= best_val, f'Best row not better than mean rates for {m}, {df_mean_rates[m].min()} vs {best_val}'
                else:
                    print(f'No mean rates for {m}')
            new_val = '\\textbf{' + df_tex[col_renaming_dict[m]].loc[best_row] + '}'
            df_tex.at[best_row, col_renaming_dict[m]] = new_val
    
    if add_mean_rates:
        for m in df_mean_rates.columns:
            df_mean_rates[m] = df_mean_rates[m] * scale_dict[m]
            if decimals_dict[m] == 2:
                df_mean_rates[m] = df_mean_rates[m].apply(lambda x: f'{x:.2f}')
            elif decimals_dict[m] == 1:
                df_mean_rates[m] = df_mean_rates[m].apply(lambda x: f'{x:.1f}')
            else:
                assert False, f'Unexpected number of decimals: {decimals_dict[m]}'
        df_mean_rates = df_mean_rates.rename(columns=col_renaming_dict)
        cols_tex = df_tex.columns
        df_tex = pd.concat([df_mean_rates, df_tex], axis=0)
        df_tex = df_tex.reset_index(drop=True)
        ## restore order columns:
        df_tex = df_tex[cols_tex]

    df_tex = df_tex.rename(columns=dict_rename_hparams)
    for c in df_tex.columns:
        if df_tex[c].dtype == 'float64' or df_tex[c].dtype == 'float32':
            df_tex[c] = df_tex[c].apply(lambda x: str(x))
        if c == 'Model':
            elements_rename_dict = {'imagenet': 'ImageNet', 'None': 'Random', 'seco': 'SeCo', np.nan: 'Mean rate'}
            df_tex[c] = df_tex[c].apply(lambda x: elements_rename_dict[x])

    if len(drop_columns_tex) > 0:
        df_tex = df_tex.drop(columns=drop_columns_tex)

    if save_table:
        assert filename is not None, 'Filename not specified'
        assert os.path.exists(folder_save), f'Folder {folder_save} does not exist'
        assert filename.endswith('.tex'), f'Filename {filename} does not end with .tex'
        path_save = os.path.join(folder_save, filename)
        df_tex.to_latex(path_save, index=False, escape=False, na_rep='N/A',
                caption=caption_tex, label=label_tex, position=position_tex)

    return df_num_val, df_tex

def plot_val_timeseries(list_ts, ax=None, metric_show='val_top_10_acc', n_epochs_expected=51):
    df_hparams_unique, df_hparams_ident, dict_metrics = create_df_val_timeseries(list_ts)
    
    cols_drop = ['time_created', 'timestamp', 'seed_used']
    df_hparams_unique = df_hparams_unique.drop(columns=cols_drop)
    cols_hparams_unique = list(df_hparams_unique.columns)
    assert len(cols_hparams_unique) == 1, f'Expected 1 column, got {len(cols_hparams_unique)}'
    cols_hparams_unique = cols_hparams_unique[0]
    cols_metrics = dict_metrics[list(dict_metrics.keys())[0]].columns
    assert metric_show in cols_metrics, f'Metric {metric_show} not in dict_metrics'

    n_datapoints = len(list_ts) * n_epochs_expected
    dict_data = {**{'epoch': np.zeros(n_datapoints, dtype=int)},
                 **{x: np.zeros(n_datapoints) for x in df_hparams_unique.columns},
                 **{x: np.zeros(n_datapoints) for x in cols_metrics}}
    
    for i, (ts, df_metrics) in enumerate(dict_metrics.items()):
        assert len(df_metrics) == n_epochs_expected, f'Expected {n_epochs_expected} epochs, got {len(df_metrics)}'
        assert (cols_metrics == df_metrics.columns).all(), 'Metrics not consistent' 
        start_ind = i * n_epochs_expected
        end_ind = (i + 1) * n_epochs_expected
        for c in df_hparams_unique.columns:
            dict_data[c][start_ind:end_ind] = df_hparams_unique[c].iloc[i]
        for c in cols_metrics:
            dict_data[c][start_ind:end_ind] = df_metrics[c].values
        dict_data['epoch'][start_ind:end_ind] = np.arange(n_epochs_expected)

    df_plot = pd.DataFrame(dict_data)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.lineplot(data=df_plot, x='epoch', y=metric_show, ax=ax, errorbar=('ci', 95),
                 hue=cols_hparams_unique, palette='tab10')
    
    return df_plot


def print_table_batchsize(split_use='test'):

    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=np.arange(199, 217)), split_use=split_use)

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different batch sizes. ' \
              'The best performing model for each metric is highlighted in bold.'

    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                           save_table=True, filename='tab_batch-size.tex', highlight_best_row=True,
                           split_use=split_use,
                           label_tex='tab:batch_size', caption_tex=caption)
    
    return (df_num_val, df_tex)

def print_table_alpha(split_use='test'):
    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=np.arange(164, 171)), split_use=split_use)

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different values of the ' + r"$\alpha$" + ' ratio loss hyperparameter. ' \
              'The best performing model for each metric is highlighted in bold.'
    
    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                            save_table=True, filename='tab_alpha-ratio.tex', highlight_best_row=True,
                           split_use=split_use,
                            label_tex='tab:alpha_ratio', caption_tex=caption)
    return (df_num_val, df_tex)

def print_table_dropout(save_table=True, include_contrastive_reg=False, split_use='test'):
    arr_inds = list(np.arange(271, 307))
    if include_contrastive_reg is False:  # structure: 12 per seed. 6 for each dropout rate. first 6 are alpha=0, second 6 are alpha=0.1
        arr_inds = arr_inds[:6] + arr_inds[12:18] + arr_inds[24:30]

    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=arr_inds), split_use=split_use)

    if include_contrastive_reg:
        assert False, 'change caption'
    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different dropout rates of the $\mathbf{z}$ embedding layer. ' \
              'Performance is stated for the prediction model without contrastive regularisation. ' 
    
    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                            save_table=save_table, filename='tab_dropout.tex', highlight_best_row=True,
                           split_use=split_use,
                            label_tex='tab:dropout', caption_tex=caption)
    return (df_num_val, df_tex)

def print_table_mlplayers_pretrained(save_table=True, split_use='test'):
    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=np.arange(324, 342)), split_use='test')

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different pretrained networks and varying number of MLP prediction layers. ' \
            'The best performing model for each metric is highlighted in bold.'

    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                                                    split_use='test', save_table=save_table, add_mean_rates=True,
                                                    filename='tab_mlp-layer_pretrained.tex', highlight_best_row=True,
                                                label_tex='tab:mlp-layer_pretrained', caption_tex=caption)
    return (df_num_val, df_tex)

def print_table_mlplayers_pretrained_lr(save_table=True, split_use='test'):
    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=np.arange(324, 360)), split_use='test')

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different pretrained networks, LR and varying number of MLP prediction layers.  ' \
            'The best performing model for each metric is highlighted in bold.'

    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                                                    split_use='test', save_table=save_table, 
                                                    filename='tab_mlp-layer_pretrained_lr.tex', highlight_best_row=True,
                                                label_tex='tab:mlp-layer_pretrained_lr', caption_tex=caption)
    return (df_num_val, df_tex)

def print_table_cr(save_table=False, split_use='test'):
    list_vnums = list(np.arange(361, 373)) + [335, 338, 341]# + list(np.arange(367, 374))
    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=list_vnums), split_use='test')

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different pretrained networks, LR and varying number of MLP prediction layers.  ' \
            'The best performing model for each metric is highlighted in bold.'

    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                                                    split_use='test', save_table=save_table, 
                                                    filename='tab_mlp-layer_pretrained_lr.tex', highlight_best_row=True,
                                                label_tex='tab:mlp-layer_pretrained_lr', caption_tex=caption)
    return (df_num_val, df_tex)

def print_table_cr_32(save_table=False, split_use='test'):
    list_vnums = list(np.arange(386, 405))
    list_vnums.remove(389)  # broken run
    list_vnums = list_vnums + [335, 338, 341]  # alpha=0 runs
    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=list_vnums), split_use='test')

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for networks with and without contrastive regularisation, '\
              'for various hyperparameter settings.'
     
    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                                                    split_use='test', save_table=save_table, 
                                                    filename='tab_cr_32.tex', highlight_best_row=True,
                                                label_tex='tab:cr_32', caption_tex=caption,
                                                drop_columns_tex=['training_method', 'Hard labels', 'name_train_loss'])
    return (df_num_val, df_tex)

def print_table_test(save_table=False, split_use='test'):
    # list_vnums = list(np.arange(361, 373)) + [335, 338, 341]# + list(np.arange(367, 374))
    # list_vnums = np.arange(380, 386)
    # list_vnums = np.concatenate((np.arange(361, 373), np.arange(380, 386), [335, 338, 341]))

    list_vnums = list(np.arange(386, 405))
    list_vnums.remove(389)
    list_vnums = list_vnums + [335, 338, 341]
    tmp_df, tmp_details = create_df_list_timestamps(list_ts=get_list_timestamps_from_vnums(
        list_vnums=list_vnums), split_use='test')

    caption = 'Mean and standard error of the mean (SEM) of validation metrics for different pretrained networks, LR and varying number of MLP prediction layers.  ' \
            'The best performing model for each metric is highlighted in bold.'

    df_num_val, df_tex = create_printable_table(df=tmp_df, hparams_use=tmp_details[1], metrics_use=tmp_details[2],
                                                    split_use='test', save_table=save_table, 
                                                    filename='tab_mlp-layer_pretrained_lr.tex', highlight_best_row=True,
                                                label_tex='tab:mlp-layer_pretrained_lr', caption_tex=caption)
    return (df_num_val, df_tex)

def print_table_results_per_species(ds, filepath_train_val_split, save_table=False,
                                    filename='tab_species_details.tex', 
                                    folder_save=os.path.join(path_dict_pecl['repo'], 'tables/'),
                                    position_tex='h'):
    
    train_ds, val_ds, test_ds = ds.split_into_train_val(filepath=filepath_train_val_split)
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(ds), 'Split not correct'
    dict_indices = {'train': train_ds.indices, 'val': val_ds.indices, 'test': test_ds.indices}
    train_ds = train_ds.dataset
    val_ds = val_ds.dataset
    test_ds = test_ds.dataset

    dict_table = {
        'species ID': np.arange(len(ds.species_list)),
        'species': [r"\textit{" + x + r"}" for x in ds.species_list]}

    for name_ds, inds in dict_indices.items():
        print(f'Number of samples in {name_ds} set: {len(inds)}')
        df_pres = ds.df_presence.iloc[inds]
        presence_vec = df_pres[ds.species_list].mean()
        dict_table[f'P(presence) {name_ds} [\%]'] = [f'{x:.2f}' for x in presence_vec.values * 100]
        # print(f'Number of species in {name_ds} set: {len(df_pres.columns)}')

    df_overview = pd.DataFrame(dict_table)
    df_overview.round(1)

    caption_tex = 'Overview of the presence of species in the training, validation and test sets. '
    label_tex = 'tab:species_details'

    if save_table:
        assert filename is not None, 'Filename not specified'
        assert os.path.exists(folder_save), f'Folder {folder_save} does not exist'
        assert filename.endswith('.tex'), f'Filename {filename} does not end with .tex'
        path_save = os.path.join(folder_save, filename)
        df_overview.to_latex(path_save, index=False, escape=False, na_rep='NaN',
                caption=caption_tex, label=label_tex, position=position_tex)

    return df_overview
