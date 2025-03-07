## script to train models 

import os, sys 
import paired_embeddings_models as pem
import loadpaths_pecl
import numpy as np
path_dict_pecl = loadpaths_pecl.loadpaths()
USE_MPS = False

def sample_random_hparams():
    ## Hyperparameters to search over:
    lr = float(10 ** np.random.uniform(-4, -3))
    batch_size = int(np.random.choice([256]))
    # upper_lim_knn = np.minimum(5, batch_size - 1)
    pecl_knn = int(np.random.randint(1, 10))
    alpha_ratio_loss = float(10 ** np.random.uniform(-1.5, -0.5))
    temperature = float(np.random.uniform(0.5, 1))
    p_dropout = float(np.random.choice([0, 0.25, 0.5]))
    k_bottom = int(np.random.choice([16, 32, 64]))
    
    hyperparams = {
            'lr': lr,
            'batch_size': batch_size,
            'pecl_knn': pecl_knn,
            'alpha_ratio_loss': alpha_ratio_loss,
            'temperature': temperature,
            'p_dropout': p_dropout,
            'k_bottom': k_bottom
        }
    
    list_variables = list(hyperparams.keys())

    ## Constant hyperparameters:
    hyperparams['freeze_resnet'] = True
    hyperparams['species_process'] = 'all'
    hyperparams['n_enc_channels'] = 256 
    hyperparams['n_layers_mlp_pred'] = 3
    hyperparams['pred_train_loss'] = 'bce'
    hyperparams['pretrained_resnet'] = 'imagenet'
    hyperparams['pecl_knn_hard_labels'] = False
    hyperparams['training_method'] = 'pred_and_pecl'
    hyperparams['use_class_weights'] = False
    hyperparams['pecl_distance_metric'] = 'softmax'
    hyperparams['n_epochs_max'] = 50
    hyperparams['n_layers_mlp_resnet'] = 1
    hyperparams['use_lr_scheduler'] = False
    hyperparams['normalise_embedding'] = 'l2'
    hyperparams['save_stats'] = True

    return hyperparams, list_variables

if __name__ == '__main__':
    ## Settings:
    bool_save_full_model = False
    bool_stop_early = True
    list_seeds_model = [42, 17, 86]
    n_combinations = 100
    eval_test_set = True
    
    ## Create all combinations of hyperparameters:
    print('Combinations will be run in this order:\n---------')
    dict_hparams = {}
    for i in range(n_combinations):
        hyperparams, list_variables = sample_random_hparams()
        dict_hparams[i] = hyperparams
        hparams_print = {k: hyperparams[k] for k in list_variables}
        print(f'- iteration {i + 1} (for {len(list_seeds_model)} seeds): {hparams_print}')
    print('-------------------\n\n')

    list_vnums = []
    n_runs = len(list_seeds_model) * n_combinations
    i_it = 0
    for _, hyperparams in dict_hparams.items():
        
        print(f'---- {i_it + 1}/{n_runs} ----')
        print({k: hyperparams[k] for k in list_variables})
        print('-------------------')

        hyperparams['save_model'] = bool_save_full_model
        hyperparams['stop_early'] = bool_stop_early
        hyperparams['tb_log_folder'] = os.path.join(path_dict_pecl['model_folder'], 'random_search/')
        hyperparams['eval_test_set'] = eval_test_set
        hyperparams['dataset_name'] = 's2bms'
        hyperparams['use_mps'] = USE_MPS
    
        for seed in list_seeds_model:
            print(f'---- {i_it + 1}/{n_runs} (seed {seed}) ----')
            hyperparams['fix_seed'] = seed
            tmp_model, _ = pem.train_pecl(**hyperparams)

            i_it += 1
            list_vnums.append(tmp_model.v_num)

        

    print('All combinations have been run in this order:\n---------')
    for i_it, hyperparams in dict_hparams.items():
        hparams_print = {k: hyperparams[k] for k in list_variables}
        for i in range(len(list_seeds_model)):
            ind_vnum = i_it * len(list_seeds_model) + i
            print(f'- {list_vnums[ind_vnum]}, iteration {ind_vnum + 1}: {hparams_print}')
    print('-------------------\n\n')