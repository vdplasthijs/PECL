## script to train models 

import os, sys 
import paired_embeddings_models as pem
import itertools
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()

if __name__ == '__main__':
    ## Settings:
    bool_save_full_model = True
    bool_stop_early = True

    ## Hyperparameters to search over:
    training_method = ['pred_and_pecl']
    species_process = ['all']
    lr = [1e-3]
    batch_size = [64] 
    pecl_knn = [4]
    pecl_knn_hard_labels = [False]
    pred_train_loss = ['bce']
    pretrained_resnet = ['seco']
    n_enc_channels = [256] 
    fix_seed = [42]
    alpha_ratio_loss = [0.1]
    freeze_resnet = [True]
    p_dropout = [0]
    n_layers_mlp_pred = [3]

    ## Create all combinations of hyperparameters:
    iterator = list(itertools.product(training_method, species_process, 
                                  n_enc_channels, lr, batch_size, pecl_knn, 
                                  pecl_knn_hard_labels, pred_train_loss, 
                                  pretrained_resnet, fix_seed, alpha_ratio_loss,
                                  freeze_resnet, p_dropout, n_layers_mlp_pred))
    n_combinations = len(iterator)
    print('Combinations will be run in this order:\n---------')
    for i, args in enumerate(iterator):
        print(f'- iteration {i + 1}: {args}')
    print('-------------------\n\n')

    i_it = 0
    list_vnums = []
    print(f'Number of combinations: {n_combinations}')
    for args in iterator:
        i_it += 1
        hyperparams = {
            'training_method': args[0],
            'species_process': args[1],
            'n_enc_channels': args[2],
            'lr': args[3],
            'batch_size': args[4],
            'pecl_knn': args[5],
            'pecl_knn_hard_labels': args[6],
            'pred_train_loss': args[7],
            'pretrained_resnet': args[8],
            'fix_seed': args[9],
            'alpha_ratio_loss': args[10],
            'freeze_resnet': args[11],
            'p_dropout': args[12],
            'n_layers_mlp_pred': args[13],
        }

        print(f'---- {i_it}/{n_combinations} ----')
        print(hyperparams)
        print('-------------------')

        ## Constant hyperparameters:
        hyperparams['use_class_weights'] = True
        hyperparams['pecl_distance_metric'] = 'softmax'
        hyperparams['n_epochs_max'] = 2
        hyperparams['n_layers_mlp_resnet'] = 1
        hyperparams['use_lr_scheduler'] = False
        hyperparams['normalise_embedding'] = 'l2'
        hyperparams['save_model'] = bool_save_full_model
        hyperparams['save_stats'] = True
        hyperparams['stop_early'] = bool_stop_early
        hyperparams['dataset_name'] = 'satbird-kenya'
        if hyperparams['dataset_name'] == 's2bms':
            filepath_train_val_split = os.path.join(path_dict_pecl['repo'], 'content/split_indices_s2bms_2024-08-14-1459.pth')
        elif hyperparams['dataset_name'] == 'satbird-kenya':
            filepath_train_val_split = os.path.join(path_dict_pecl['repo'],'content/split_indices_Kenya_2024-08-14-1506.pth')
        elif hyperparams['dataset_name'] == 'satbird-usawinter':
            filepath_train_val_split = os.path.join(path_dict_pecl['repo'],'content/split_indices_USA_winter_2024-08-14-1506.pth')

        # if i_it <= 15:
        #     continue

        tmp_model, _ = pem.train_pecl(**hyperparams)
        list_vnums.append(tmp_model.v_num)

    print('All combinations have been run in this order:\n---------')
    for i, args in enumerate(iterator):
        print(f'- {list_vnums[i]}, iteration {i + 1}: {args}')
    print('-------------------\n\n')