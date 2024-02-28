## script to train models 

import os, sys 
import paired_embeddings_models as pem
import itertools

if __name__ == '__main__':
    ## Hyperparameters to search over:
    training_method = ['pred_incl_enc']
    species_process = ['all']
    lr = [1e-3]
    batch_size = [64]
    pecl_knn = [5]
    pecl_knn_hard_labels = [True]
    pred_train_loss = ['weighted-bce', 'bce']
    pretrained_resnet = ['seco']
    n_enc_channels = [256] 
    fix_seed = [42]
    
    ## Create all combinations of hyperparameters:
    iterator = list(itertools.product(training_method, species_process, 
                                  n_enc_channels, lr, batch_size, pecl_knn, 
                                  pecl_knn_hard_labels, pred_train_loss, 
                                  pretrained_resnet, fix_seed))
    n_combinations = len(iterator)
    i_it = 0
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
            'fix_seed': args[9]
        }

        print(f'---- {i_it}/{n_combinations} ----')
        print(hyperparams)
        print('-------------------')

        ## Constant hyperparameters:
        hyperparams['use_class_weights'] = True
        hyperparams['pecl_distance_metric'] = 'softmax'
        hyperparams['n_epochs_max'] = 100
        hyperparams['freeze_resnet'] = True
        hyperparams['n_layers_mlp_resnet'] = 1
        hyperparams['n_layers_mlp_pred'] = 2
        hyperparams['use_lr_scheduler'] = True
        hyperparams['normalise_embedding'] = 'l2'
        hyperparams['save_model'] = True
        hyperparams['save_stats'] = True

        tmp_model, _ = pem.train_pecl(**hyperparams)