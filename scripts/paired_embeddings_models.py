import os, sys, copy
import numpy as np
from tqdm import tqdm
import datetime, pickle
import pickle
import pandas as pd 
import shapely.geometry
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import transforms
# import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()
import create_dataset_utils as cdu 
from load_seco_resnet import map_seco_to_torchvision_weights
from DataSetImagePresence import DataSetImagePresence

class ImageEncoder(pl.LightningModule):
    '''
    Encode image using CNN Resnet + FCN.
    Train/test using various methods: PECL, or direct prediction.

    https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
    Partly inspired by https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
    '''
    resnets = {
        18: models.resnet18,
        # 34: models.resnet34,
        50: models.resnet50,
        # 101: models.resnet101,
        # 152: models.resnet152,
    }

    def __init__(self, n_species=62, n_enc_channels=128, n_bands=4, 
                 n_layers_mlp_resnet=1, n_layers_mlp_pred=2,
                 pretrained_resnet='imagenet', freeze_resnet=True,
                 optimizer_name='Adam', resnet_version=18,
                 pecl_distance_metric='softmax',
                 pred_train_loss='mse', class_weights=None,
                 lr=1e-3, pecl_knn=5, pecl_knn_hard_labels=False,
                 training_method='pecl', alpha_ratio_loss=None,
                 normalise_embedding='l2', use_mps=True,
                 use_lr_scheduler=False, seed_used=None, 
                 batch_size_used=None, p_dropout=0,
                 temperature=0.5,
                 verbose=1, time_created=None):
        super(ImageEncoder, self).__init__()
        self.save_hyperparameters()
        self.n_species = n_species
        self.n_enc_channels = n_enc_channels
        self.n_bands = n_bands
        assert self.n_bands in [3, 4], f'Number of bands {self.n_bands} not implemented.'
        self.verbose = verbose
        self.seed_used = seed_used  # Save seed used in training function
        self.batch_size_used = batch_size_used  ## only saved for reference, not required
        self.pretrained_resnet = pretrained_resnet
        self.freeze_resnet = freeze_resnet
        self.lr = lr
        self.alpha_ratio_loss = alpha_ratio_loss
        self.resnet_version = resnet_version
        self.n_layers_mlp_resnet = n_layers_mlp_resnet
        assert self.n_layers_mlp_resnet == 1, 'Expecting 1 layer MLP for projection head for now.'
        # assert self.n_layers_mlp_resnet in [1, 2], f'Number of MLP layers {self.n_layers_mlp_resnet} not implemented.'
        self.n_layers_mlp_pred = n_layers_mlp_pred
        assert self.n_layers_mlp_pred in [1, 2, 3], 'Expecting 1 or 2 or 3 layer MLP for prediction head for now.'
        self.pecl_distance_metric = pecl_distance_metric
        self.optimizer_name = optimizer_name
        self.use_mps = use_mps
        self.normalise_embedding = normalise_embedding
        assert self.normalise_embedding == 'l2', 'Currently expecting l2 normalisation'
        self.pecl_knn = pecl_knn
        self.pecl_knn_hard_labels = pecl_knn_hard_labels
        self.use_lr_scheduler = use_lr_scheduler
        self.use_dropout = p_dropout > 0
        self.p_dropout = p_dropout
        self.temperature = temperature
        if self.use_dropout:
            print('Using dropout.')
        self.model_name = None
        self.v_num = None
        self.log_dir = None
        self.time_created = time_created
        self.description = f'ImageEncoder with {n_enc_channels} encoding channels, {n_bands} bands, {n_species} species, {n_layers_mlp_resnet} MLP layers, {resnet_version} Resnet, {pecl_distance_metric} distance metric, {training_method} training method.'
        self.df_metrics = None 
        self.test_metrics = None 
        self.build_class_weights(class_weights=class_weights)
        self.build_model()
        self.build_training_method(training_method=training_method, pred_train_loss=pred_train_loss)
       
        if self.pred_train_loss in ['weighted-bce', 'weighted-ce']:
            assert self.class_weights is not None, 'Class weights not set.'

    def __str__(self) -> str:
        return super().__str__() + f' {self.description}'
    
    def __repr__(self) -> str:
        return super().__repr__() + f' {self.description}'

    def change_description(self, new_description='', add=True):
        '''Just used for keeping notes etc.'''
        if add:
            self.description = self.description + '\n' + new_description
        else:
            self.description = new_description

    def build_class_weights(self, class_weights):
        ## Load class weights
        if class_weights is not None:
            assert class_weights.ndim == 1, f'Class weights shape {class_weights.shape} not 1D.'
            assert class_weights.shape[0] == self.n_species, f'Class weights shape {class_weights.shape} does not match number of species {self.n_species}.'
            self.class_weights = torch.tensor(class_weights).float() 
            print(f'Loaded {self.class_weights.shape[0]} class weights on {self.class_weights.device}.')
            if self.use_mps:
                self.class_weights = self.class_weights.to('mps')
            else:
                self.class_weights = self.class_weights.to('cuda')
            print(f'Class weights now on {self.class_weights.device}.')
        else:
            print('No class weights.')
            self.class_weights = None

        ## Some checks:
        if self.class_weights is not None:
            assert self.class_weights is not None, 'Class weights not set.'
            assert type(self.class_weights) == torch.Tensor, f'Class weights type {type(self.class_weights)} not torch.Tensor.'
            assert self.class_weights.ndim == 1, f'Class weights shape {self.class_weights.shape} not 1D.'
            assert self.class_weights.shape[0] == self.n_species, f'Class weights shape {self.class_weights.shape} does not match number of species {self.n_species}.'
        
    def build_training_method(self, training_method, pred_train_loss):
        self.train_im_enc_during_pred = False
        if training_method == 'pecl':
            self.forward_pass = self.pecl_pass
            self.pred_train_loss = None
            self.name_train_loss = f'pecl-{self.pecl_distance_metric}'
            self.freeze_resnet_layers(freeze_all_but_last=self.freeze_resnet,
                                      freeze_last=False)
            self.freeze_prediction_model(freeze=True)
        elif training_method == 'pred':
            self.forward_pass = self.pred_pass
            self.pred_train_loss = pred_train_loss
            self.name_train_loss = f'pred-{pred_train_loss}'
            self.freeze_resnet_layers(freeze_all_but_last=True,
                                      freeze_last=True)
            self.freeze_prediction_model(freeze=False)
        elif training_method == 'pred_incl_enc':
            self.forward_pass = self.pred_pass
            self.pred_train_loss = pred_train_loss
            self.name_train_loss = f'pred-{pred_train_loss}'
            self.train_im_enc_during_pred = True
            self.freeze_resnet_layers(freeze_all_but_last=self.freeze_resnet,
                                      freeze_last=False)
            self.freeze_prediction_model(freeze=False)
        elif training_method == 'pred_and_pecl':
            self.forward_pass = self.pred_and_pecl_pass
            self.pred_train_loss = pred_train_loss
            self.name_train_loss = f'pred-{pred_train_loss}_pecl-{self.pecl_distance_metric}'
            assert self.alpha_ratio_loss is not None, 'Expecting alpha_ratio_loss to be set.'
            assert self.alpha_ratio_loss >= 0, 'Expecting alpha_ratio_loss to be >= 0.'    
            
            self.freeze_resnet_layers(freeze_all_but_last=self.freeze_resnet,
                                      freeze_last=False)
            self.freeze_prediction_model(freeze=False)    
        else:
            assert False, f'Training method {training_method} not implemented.'

    def build_model(self):
        ## Load Resnet, if needed modify first layer to accept 4 bands
        if self.pretrained_resnet == 'seco':
            self.resnet = map_seco_to_torchvision_weights(model=None, device_use='mps' if self.use_mps else 'gpu',
                                                          resnet_name=f'resnet{self.resnet_version}', verbose=0)
            self.pretrained_weights_name = f'seco_resnet{self.resnet_version}_1m'
            print('Loaded Resnet with SeCo weights.')
        elif self.pretrained_resnet is None or self.pretrained_resnet == False:
            self.resnet = self.resnets[self.resnet_version](weights=None)
            print(f'Loaded Resnet{self.resnet_version} with random weights.')
        elif self.pretrained_resnet == 'imagenet' or self.pretrained_resnet == True:
            self.pretrained_weights_name = "IMAGENET1K_V1"
            self.resnet = self.resnets[self.resnet_version](weights=self.pretrained_weights_name)
            print(f'Loaded Resnet{self.resnet_version} with {self.pretrained_weights_name} weights.')
        else:
            raise ValueError(f'Pretrained resnet {self.pretrained_resnet} not implemented.')
        
        if self.n_bands == 3:
            pass 
        elif self.n_bands == 4:  # https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py
            weight = self.resnet.conv1.weight.clone()  # copy the weights from the first layer
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # change the first layer to accept 4 channels
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3] = weight
                self.resnet.conv1.weight[:, 3] = weight[:, 0]
        else:
            assert False, f'Number of bands {self.n_bands} not implemented.'

        ## Modify last layer to output n_enc_channels
        if self.n_layers_mlp_resnet == 1:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.n_enc_channels)
        elif self.n_layers_mlp_resnet == 2:
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_enc_channels)
            )
        else:
            assert False, f'Number of layers {self.n_layers_mlp_resnet} not implemented.'    

        ## Prediction model to predict presence/absence from encoded image
        if self.n_layers_mlp_pred == 1:
            self.prediction_model = nn.Sequential(
                nn.Linear(self.n_enc_channels, self.n_species),
                nn.Sigmoid())
        elif self.n_layers_mlp_pred == 2:
            self.prediction_model = nn.Sequential(
                nn.Linear(self.n_enc_channels, self.n_enc_channels),
                nn.ReLU(),
                nn.Linear(self.n_enc_channels, self.n_species),
                nn.Sigmoid())
        elif self.n_layers_mlp_pred == 3:
            self.prediction_model = nn.Sequential(
                nn.Linear(self.n_enc_channels, self.n_enc_channels),
                nn.ReLU(),
                nn.Linear(self.n_enc_channels, self.n_enc_channels),
                nn.ReLU(),
                nn.Linear(self.n_enc_channels, self.n_species),
                nn.Sigmoid())
        else:
            raise ValueError(f'Number of layers {self.n_layers_mlp_pred} not implemented.')

    def freeze_resnet_layers(self, freeze_all_but_last=True, freeze_last=False):
        layers_resnet = list(self.resnet.children())
        n_layers = len(layers_resnet)
        for i_c, child in enumerate(layers_resnet):
            if i_c < n_layers - 1:  # everything except last layer , which is the FC layer
                for param in child.parameters():
                    if freeze_all_but_last:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for param in child.parameters():
                    if freeze_last:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        print(f'Freezing all but last layer: {freeze_all_but_last}, last layer: {freeze_last}.')

    def freeze_prediction_model(self, freeze=True):
        for param in self.prediction_model.parameters():
            param.requires_grad = not freeze
        print(f'Freezing prediction model: {freeze}.')

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        encoding = self.resnet(x)
        if self.normalise_embedding is None:
            assert False, 'Expecting normalisation of embedding.'
        elif self.normalise_embedding == 'l2':
            encoding = F.normalize(encoding, p=2, dim=1)  ## dim=0; normalise each feature element across batch. dim=1; normalise each batch element across features.
        else:
            assert False, f'Normalisation method {self.normalise_embedding} not implemented.'
        
        if self.use_dropout:
            encoding = F.dropout(encoding, p=self.p_dropout, training=self.training)

        return encoding
    
    def configure_optimizers(self):
        if self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            assert False, f'Optimizer {self.optimizer_name} not implemented.'

        if self.use_lr_scheduler:
            print('Using ReduceLROnPlateau scheduler.')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                             factor=0.1, patience=10, verbose=True)
            return {"optimizer": self.optimizer, 
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        else:
            return self.optimizer
    
    def pecl_pass(self, batch):
        '''Train the encoding model using the PECL method.'''
        im, pres_vec = batch
        
        # Forward pass
        im_enc = self.forward(im)
        if self.normalise_embedding == 'l2':
            pres_vec = F.normalize(pres_vec, p=2, dim=1)

        '''
        Maybe this should be split up.. Forward() should do the im_enc & pred.
        If pred not needed, forward() just doesnt do it and returns None. 

        Then this function just becomes the CL loss function.
        Another becomes loss for prediction. 

        Then train/val/test step just calls forward() and the loss function.
        
        '''
        loss, im_enc = self.pecl_loss(im_enc, pres_vec)
        return loss, None, im_enc

    def pecl_loss(self, im_enc, pres_vec):
        if im_enc.shape[0] == 1:
            print('Batch size 1.')
            return 0, im_enc
        if self.pecl_distance_metric == 'softmax':
            if self.pecl_knn is not None:
                flatten_dist = False
            else:
                flatten_dist = True
            # assert pres_vec.shape[0] >= self.pecl_knn, f'Batch size {pres_vec.shape[0]} must be >= knn {self.pecl_knn}.'   
            dist_array_ims = normalised_softmax_distance_batch(im_enc, flatten=flatten_dist, temperature=self.temperature)
            dist_array_pres = normalised_softmax_distance_batch(pres_vec, flatten=flatten_dist, knn=self.pecl_knn,
                                                                knn_hard_labels=self.pecl_knn_hard_labels,
                                                                temperature=self.temperature,
                                                                similarity_function='cosine',
                                                                soft_weights_squared=True,  # only matters if hard labels is False
                                                                inner_prod_only=True)
                                                                
            inds_one = torch.where(dist_array_pres > 0)
            dist_array_pres = dist_array_pres[inds_one]
            ## cross entropy loss
            assert (dist_array_pres > 0).any(), (dist_array_pres, im_enc.shape, pres_vec.shape, dist_array_ims.shape, dist_array_ims)

            dist_array_ims = dist_array_ims[inds_one]
            assert (dist_array_ims >= 0).all(), (dist_array_ims, im_enc)
            assert (dist_array_ims <= 1.01).all(), (dist_array_ims, im_enc)
            assert (dist_array_pres >= 0).all(), (dist_array_pres, pres_vec)
            assert (dist_array_pres <= 1.01).all(), (dist_array_pres, pres_vec, torch.where(dist_array_pres > 1), dist_array_pres[torch.where(dist_array_pres > 1)])  # changed to 1.01 because of numerical issues when vals = 1.000
            loss = (-1 * torch.log(dist_array_ims + 1e-8) * dist_array_pres).mean()
            assert loss >= 0, loss
        else:
            assert False, f'Distance metric {self.pecl_distance_metric} deprecated.'
            dist_array_ims = calculate_similarity_batch(im_enc, distance_metric=self.pecl_distance_metric)
            dist_array_pres = calculate_similarity_batch(pres_vec, distance_metric=self.pecl_distance_metric)
            
            loss_array = torch.abs(dist_array_ims - dist_array_pres)  # L1 loss
            loss = torch.mean(loss_array)

        return loss, im_enc

    def pred_pass(self, batch):
        '''Only trains the prediction model, not the encoding model.'''
        im, pres_vec = batch
        
        # Forward pass
        if self.train_im_enc_during_pred:
            im_enc = self.forward(im)
        else:
            with torch.no_grad():  # Don't train encoding model here. 
                im_enc = self.forward(im)
        pres_pred = self.prediction_model(im_enc)
        loss = self.pred_loss(pres_pred, pres_vec)
        return loss, None, im_enc
    
    def pred_and_pecl_pass(self, batch):
        '''Train the encoding model using the PECL method, and the prediction model.'''
        im, pres_vec = batch
        
        # Forward pass
        im_enc = self.forward(im)
        if self.normalise_embedding == 'l2':
            pres_vec_pecl = F.normalize(pres_vec, p=2, dim=1)
        else:
            assert False, f'Normalisation method {self.normalise_embedding} not implemented.'

        pres_pred = self.prediction_model(im_enc)
        loss_pred = self.pred_loss(pres_pred, pres_vec)
        if self.alpha_ratio_loss > 0:
            loss_pecl, im_enc = self.pecl_loss(im_enc, pres_vec_pecl)
            loss = loss_pred + self.alpha_ratio_loss * loss_pecl
        else:
            loss = loss_pred
            loss_pecl = 0
        return loss, (loss_pred, self.alpha_ratio_loss * loss_pecl), im_enc

    def pred_loss(self, pres_pred, pres_vec):
        if self.pred_train_loss == 'mse':
            loss = F.mse_loss(pres_pred, pres_vec)
        elif self.pred_train_loss == 'mae':
            loss = nn.L1Loss()(pres_pred, pres_vec)
        elif self.pred_train_loss == 'ce':
            loss = nn.CrossEntropyLoss(reduction='mean')(pres_pred, pres_vec)
        elif self.pred_train_loss == 'bce':
            loss = nn.BCELoss(reduction='mean')(pres_pred, pres_vec)
        elif self.pred_train_loss == 'weighted-ce' and self.class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')(pres_pred, pres_vec)
        elif self.pred_train_loss == 'weighted-bce' and self.class_weights is not None:
            loss = self.weighted_bce_loss(pres_pred, pres_vec)
        else:
            assert False, f'Prediction training loss {self.pred_train_loss} not implemented.'
        return loss
                
    def training_step(self, batch, batch_idx):
        loss, _, __ = self.forward_pass(batch)
        self.log(f'train_{self.name_train_loss}_loss', loss, on_epoch=True, on_step=False)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, split_loss, im_enc = self.forward_pass(batch)
            self.log(f'val_{self.name_train_loss}_loss', loss, on_epoch=True, on_step=False)  # saving name loss function used so it can be recovered later
            self.log('val_loss', loss, on_epoch=True, on_step=False)  # also save as val_loss for tensorboard (and lr_scheduler etc)

            ## Evaluate split loss:
            if split_loss is not None:
                loss_pred, loss_pecl = split_loss
                self.log(f'val_pecl-{self.pecl_distance_metric}_loss', loss_pecl, on_epoch=True, on_step=False)
                ratio = loss_pred / loss_pecl
                self.log(f'val_ratio_pred_pecl', ratio, on_epoch=True, on_step=False)

            ## Evaluate prediction:
            pres_pred = self.prediction_model(im_enc)  ## not ideal to do this here, but for now it's fine.
            pres_vec = batch[1]
            assert pres_pred.shape == pres_vec.shape, f'Shape pred {pres_pred.shape}, shape labels {pres_vec.shape}'
            distance_pred_label_means = torch.mean(torch.abs(pres_vec.mean(0) - pres_pred.mean(0)))
            self.log(f'val_distance_pred_label_means', distance_pred_label_means, on_epoch=True, on_step=False)
            for k in [1, 5, 10, 20]:
                top_k_acc = self.top_k_accuracy(preds=pres_pred, target=pres_vec, k=k)
                self.log(f'val_top_{k}_acc', top_k_acc, on_epoch=True, on_step=False)
            mse_loss = F.mse_loss(pres_pred, pres_vec)
            self.log(f'val_mse_loss', mse_loss, on_epoch=True, on_step=False)
            mae_loss = nn.L1Loss()(pres_pred, pres_vec)
            self.log(f'val_mae_loss', mae_loss, on_epoch=True, on_step=False)
            ce_loss = nn.CrossEntropyLoss(reduction='mean')(pres_pred, pres_vec)
            self.log(f'val_ce_loss', ce_loss, on_epoch=True, on_step=False)
            bce_loss = nn.BCELoss(reduction='mean')(pres_pred, pres_vec)
            self.log(f'val_bce_loss', bce_loss, on_epoch=True, on_step=False)
            if self.class_weights is not None:
                bce_weighted_loss = self.weighted_bce_loss(pres_pred, pres_vec)
                self.log(f'val_weighted-bce_loss', bce_weighted_loss, on_epoch=True, on_step=False)
                ce_weighted_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')(pres_pred, pres_vec)
                self.log(f'val_weighted-ce_loss', ce_weighted_loss, on_epoch=True, on_step=False)
            return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, split_loss, im_enc = self.forward_pass(batch)
            self.log(f'test_{self.name_train_loss}_loss', loss)  # saving name loss function used so it can be recovered later
            self.log('test_loss', loss)  # also save as test_loss for tensorboard (and lr_scheduler etc)

            ## Evaluate split loss:
            if split_loss is not None:
                loss_pred, loss_pecl = split_loss
                self.log(f'test_pecl-{self.pecl_distance_metric}_loss', loss_pecl)
                ratio = loss_pred / loss_pecl
                self.log(f'test_ratio_pred_pecl', ratio)

            ## Evaluate prediction:
            pres_pred = self.prediction_model(im_enc)  ## not ideal to do this here, but for now it's fine.
            pres_vec = batch[1]
            assert pres_pred.shape == pres_vec.shape, f'Shape pred {pres_pred.shape}, shape labels {pres_vec.shape}'
            distance_pred_label_means = torch.mean(torch.abs(pres_vec.mean(0) - pres_pred.mean(0)))
            self.log(f'test_distance_pred_label_means', distance_pred_label_means)
            for k in [1, 5, 10, 20]:
                top_k_acc = self.top_k_accuracy(preds=pres_pred, target=pres_vec, k=k)
                self.log(f'test_top_{k}_acc', top_k_acc)
            mse_loss = F.mse_loss(pres_pred, pres_vec)
            self.log(f'test_mse_loss', mse_loss)
            mae_loss = nn.L1Loss()(pres_pred, pres_vec)
            self.log(f'test_mae_loss', mae_loss)
            ce_loss = nn.CrossEntropyLoss(reduction='mean')(pres_pred, pres_vec)
            self.log(f'test_ce_loss', ce_loss)
            bce_loss = nn.BCELoss(reduction='mean')(pres_pred, pres_vec)
            self.log(f'test_bce_loss', bce_loss)
            if self.class_weights is not None:
                bce_weighted_loss = self.weighted_bce_loss(pres_pred, pres_vec)
                self.log(f'test_weighted-bce_loss', bce_weighted_loss)
                ce_weighted_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')(pres_pred, pres_vec)
                self.log(f'test_weighted-ce_loss', ce_weighted_loss)
            return loss
    
    def weighted_bce_loss(self, preds, target):
        '''Weighted binary cross entropy loss.'''
        assert preds.shape == target.shape
        intermediate_loss = nn.BCELoss(reduction='none')(preds, target)  #TODO: add mean over just batch dimension here?
        # assert intermediate_loss.shape == target.shape, (intermediate_loss.shape, target.shape)
        assert intermediate_loss.shape[1] == self.n_species, (intermediate_loss.shape, self.n_species)
        intermediate_loss = torch.mean(intermediate_loss, dim=0)  
        assert intermediate_loss.shape == self.class_weights.shape, (intermediate_loss.shape, self.class_weights.shape)
        loss = torch.mean(intermediate_loss * self.class_weights)
        return loss

    def top_k_accuracy(self, preds, target, k=1):
        '''Calculate top-k accuracy: the proportion of samples where the target class is within the top k predicted classes.'''
        assert preds.shape == target.shape
        inds_sorted_preds = torch.argsort(preds, dim=1, descending=True)  # dim =1; sort along 2nd dimension (ie per sample)
        inds_sorted_target = torch.argsort(target, dim=1, descending=True)
        len_batch = preds.shape[0]
        
        ## Calculate top-k accuracy using tmp binary vectors that are 1 for the top-k predictions
        tmp_pred_greater_th = torch.zeros_like(preds)
        tmp_target_greater_th = torch.zeros_like(target)
        for row in range(len_batch):
            tmp_pred_greater_th[row, inds_sorted_preds[row, :k]] = 1
            tmp_target_greater_th[row, inds_sorted_target[row, :k]] = 1

        assert tmp_pred_greater_th.sum() <= k * len_batch, tmp_pred_greater_th.sum() 
        assert tmp_target_greater_th.sum() <= k * len_batch, tmp_target_greater_th.sum()

        tmp_joint = tmp_pred_greater_th * tmp_target_greater_th
        n_present = torch.sum(tmp_joint, dim=1)  ## sum per batch sample

        for n in n_present:
            assert n <= k, n_present

        top_k_acc = n_present.float() / k  # accuracy per batch sample 
        top_k_acc = top_k_acc.mean()

        return top_k_acc
    
    def store_val_metrics(self, metrics):
        if self.df_metrics is not None:
            print('-- When saving metrics, df_metrics already exists. Appending to old_df_metrics.')
            if hasattr(self, 'old_df_metrics'):
                assert type(self.old_df_metrics) == list
                self.old_df_metrics.append(copy.deepcopy(self.df_metrics))
            else:
                self.old_df_metrics = [copy.deepcopy(self.df_metrics)]
            print(f'-- Old df_metrics stored in old_df_metrics list of length {len(self.old_df_metrics)}')

        self.n_epochs_converged = len(metrics) 
        metrics_float = []
        self.set_metric_names = set()
        for ii in range(len(metrics)):
            metrics_float.append({})
            for key in metrics[ii].keys():
                self.set_metric_names.add(key)
                value = metrics[ii][key]
                if type(value) == float:
                    continue 
                if type(value) == torch.Tensor:
                    metrics_float[ii][key] = value.detach().cpu().numpy()
                assert type(metrics_float[ii][key]) == np.ndarray, type(metrics_float[ii][key])
                assert metrics_float[ii][key].shape == (), metrics_float[ii][key]
                metrics_float[ii][key] = float(metrics_float[ii][key])

        self.metric_arrays = {}
        for key in self.set_metric_names:
            self.metric_arrays[key] = np.zeros(self.n_epochs_converged) + np.nan
            for ii in range(self.n_epochs_converged):
                if key in metrics_float[ii]:
                    self.metric_arrays[key][ii] = metrics_float[ii][key]
         
        self.df_metrics = pd.DataFrame(self.metric_arrays)
        return 
    
    def store_test_metrics(self, metrics):
        metrics_float = {}
        assert len(metrics) == 1, f'Expected 1 dict, but got {len(metrics)}'
        for k, v in metrics[0].items():
            if type(v) == torch.Tensor:
                metrics_float[k] = v.detach().cpu().numpy()
            else:
                metrics_float[k] = v
        self.test_metrics = pd.DataFrame(metrics_float, index=[0])

    def save_stats(self, folder=os.path.join(path_dict_pecl['model_folder'], 'stats/'),
                   verbose=1):
        '''Save logger stats & model params only. '''
        assert self.df_metrics is not None, 'Metrics not stored yet.'
        assert os.path.exists(folder), f'Folder {folder} does not exist.'
        ## Save v_num that is used for tensorboard
        if self.v_num is None:
            self.v_num = self.logger.version
        ## Save logging directory that is used for tensorboard
        if self.log_dir is None:
            self.log_dir = self.logger.log_dir

        if self.model_name is None:
            timestamp = cdu.create_timestamp()
            self.model_name = f'PECL-ImEn_{timestamp}_vnum-{self.v_num}'

        self.dict_save = {
            'hparams': {**self.hparams, **{'name_train_loss': self.name_train_loss},
                        'pred_train_loss': self.pred_train_loss,
                        'n_epochs_converged': self.n_epochs_converged},
            'v_num': self.v_num,
            'log_dir': self.log_dir,
            'logger': self.logger.__dict__,
            'df_metrics': self.df_metrics,
            'test_metrics': self.test_metrics,
        }
        self.filename = f'{self.model_name}_stats.pkl'
        self.filepath = os.path.join(folder, self.filename)
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.dict_save, f)
        if verbose > 0:
            print(f'Stats saved as {self.filename} at {self.filepath}')

    def save_model(self, folder=os.path.join(path_dict_pecl['model_folder'], 'full_models/'), 
                   verbose=1):
        '''Save model'''
        assert self.df_metrics is not None, 'Metrics not stored yet.' 
        ## Save v_num that is used for tensorboard
        if self.v_num is None:
            self.v_num = self.logger.version
        ## Save logging directory that is used for tensorboard
        if self.log_dir is None:
            self.log_dir = self.logger.log_dir
        
        if self.model_name is None:
            timestamp = cdu.create_timestamp()
            self.model_name = f'PECL-ImEn_{timestamp}_vnum-{self.v_num}'
        self.filename = f'{self.model_name}.data'
        self.filepath = os.path.join(folder, self.filename)

        file_handle = open(self.filepath, 'wb')
        pickle.dump(self, file_handle)

        if verbose > 0:
            print(f'PECL-ImEn model saved as {self.filename} at {self.filepath}')
        return self.filepath

def load_model(folder=os.path.join(path_dict_pecl['model_folder'], 'full_models/'), 
               filename='', verbose=1):
    '''Load previously saved (pickled) model'''
    assert filename != '', 'Filename not provided.'
    assert filename.endswith('.data'), f'Filename {filename} should end with .data'
    with open(os.path.join(folder, filename), 'rb') as f:
        model = pickle.load(f)

    if verbose > 0:  
        print(f'Loaded {model}')
        if hasattr(model, 'description') and verbose > 0:
            print(model.description)

    return model 

def load_model_from_ckpt(v_num=None, filepath=None, 
                         base_folder=os.path.join(path_dict_pecl['model_folder'], 'lightning_logs/')):
    '''Load model from checkpoint file.'''
    assert filepath is not None or v_num is not None, 'Version number and filepath not provided.'
    if filepath is None:
        assert v_num is not None, 'Version number and filepath not provided.'
        assert type(v_num) == int, f'Expected int for version number, but got {type(v_num)}'
        folder_v_num = os.path.join(base_folder, f'version_{v_num}')
        assert os.path.exists(folder_v_num), f'Folder {folder_v_num} does not exist.'
        folder_ckpt = os.path.join(folder_v_num, 'checkpoints')
        contents_folder_ckpt = os.listdir(folder_ckpt)
        assert len(contents_folder_ckpt) == 1, f'Expected 1 file in folder {folder_ckpt}, but got {len(contents_folder_ckpt)}'
        assert contents_folder_ckpt[0].endswith('.ckpt'), f'File {contents_folder_ckpt[0]} should end with .ckpt'
        filepath = os.path.join(folder_ckpt, contents_folder_ckpt[0])
    elif v_num is None:
        assert filepath is not None, 'Version number and filepath not provided.'
        assert filepath.endswith('.ckpt'), f'Filepath {filepath} should end with .ckpt'
        assert os.path.exists(filepath), f'File {filepath} does not exist.'
    model = ImageEncoder.load_from_checkpoint(filepath)
    return model

def load_stats(folder=None, filename=None, timestamp=None, verbose=1):
    '''Load previously saved (pickled) stats'''
    assert (filename is not None and timestamp is None) or (filename is None and timestamp is not None), 'Provide either filename or timestamp, not both.'
    if folder is None:
        folder = os.path.join(path_dict_pecl['model_folder'], 'stats/')
    
    if filename is None:
        list_files = os.listdir(folder)
        list_files = [f for f in list_files if f.endswith('.pkl')]
        list_files = [f for f in list_files if timestamp in f]
        assert len(list_files) == 1, f'Expected 1 file with timestamp {timestamp} in {folder}, but got {len(list_files)}'
        filename = list_files[0]
    assert filename != '', 'Filename not provided.'
    assert filename.endswith('.pkl'), f'Filename {filename} should end with .pkl'
    with open(os.path.join(folder, filename), 'rb') as f:
        dict_load = pickle.load(f)
    if verbose > 0:
        print(f'Loaded stats from {filename} at {folder}')
    return dict_load

def normalised_softmax_distance_batch(samples, temperature=0.5, exclude_diag_in_denominator=True,
                                      flatten=True, knn=None, knn_hard_labels=False,
                                      soft_weights_squared=True, suppress_knn_size_warning=True,
                                      similarity_function='inner', inner_prod_only=False):
    '''Calculate the distance between two embeddings using the normalised softmax distance.'''
    assert temperature > 0, f'Temperature {temperature} should be > 0.'
    assert samples.ndim == 2, f'Expected 2D tensor, but got {samples.ndim}D tensor.'  # (batch, features)
    assert (samples <= 1).all(), f'Values should be <= 1, but got {samples.max()}'
    if similarity_function == 'inner':
        pass
    elif similarity_function == 'cosine':
        samples = F.normalize(samples, p=2, dim=1)
        # assert False, 'Expected samples to be already normalised..'
    else:
        assert False, f'Similarity function {similarity_function} not implemented.'

    inner_prod_mat = torch.mm(samples, samples.t())

    if inner_prod_only is False:  # convert to softmax distance
        inner_prod_mat = inner_prod_mat / temperature
        inner_prod_mat = torch.exp(inner_prod_mat)
        sum_inner_prod_mat = torch.sum(inner_prod_mat, dim=1)
        if exclude_diag_in_denominator:
            diag = torch.diag(inner_prod_mat)
            sum_inner_prod_mat = sum_inner_prod_mat - diag
        sum_inner_prod_mat = sum_inner_prod_mat + 1e-8  # avoid division by zero
        inner_prod_mat = inner_prod_mat / sum_inner_prod_mat[:, None]

    if knn is not None:
        assert flatten is False, 'Flatten should be False if KNN is used.'
        assert type(knn) == int, f'Expected int for knn, but got {type(knn)}'
        assert knn > 0, f'Expected knn > 0, but got {knn}'
        if knn >= samples.shape[0] - 1:
            if suppress_knn_size_warning is False:
                print(f'Expected knn < number of samples - 1, but got {knn} and {samples.shape[0]}. This can happen if final batch of data loader happens to be very small. Setting k-1 to batch size for this batch only.')
            # return torch.zeros_like(inner_prod_mat)
            knn = samples.shape[0] - 1
        inner_prod_mat = inner_prod_mat - torch.diag(inner_prod_mat.diag())  # set diagonal to 0 so it doesn't get picked with KNN
        knn_inner_prod_mat = torch.zeros_like(inner_prod_mat)
        inds_positive = torch.topk(inner_prod_mat, k=knn, dim=1, largest=True, sorted=False)[1]
        if knn_hard_labels:
            knn_inner_prod_mat.scatter_(1, inds_positive, 1)
        else:
            for row, cols in enumerate(inds_positive):
                if soft_weights_squared:
                    knn_inner_prod_mat[row, cols] = inner_prod_mat[row, cols] ** 2
                else:
                    knn_inner_prod_mat[row, cols] = inner_prod_mat[row, cols]
        return knn_inner_prod_mat
    if flatten:  # only return upper triangular part because of symmetry
        inds_upper_triu = torch.triu_indices(inner_prod_mat.shape[0], inner_prod_mat.shape[1], offset=1)
        inner_prod_mat = inner_prod_mat[inds_upper_triu[0], inds_upper_triu[1]]
    return inner_prod_mat
    
def calculate_similarity_batch(samples, distance_metric='cosine'):
    curr_batch_size = len(samples)
    ## Calculate distance between pairs of images and pairs of presence vectors
    similarity_array_samples = torch.zeros(curr_batch_size * (curr_batch_size - 1) // 2)
    ind_pair = -1
    for i in range(curr_batch_size):
        for j in range(i + 1, curr_batch_size):  # avoid duplicates
            ind_pair += 1
            # print(ind_pair, i, j, samples[i].shape, samples[j].shape)
            if distance_metric == 'cosine':
                similarity_array_samples[ind_pair] = 1 - F.cosine_similarity(samples[i], samples[j], dim=0)
            elif distance_metric == 'euclidean':
                similarity_array_samples[ind_pair] = F.pairwise_distance(samples[i], samples[j], p=2)
            else:
                assert False, f'Distance metric {distance_metric} not implemented.'
    return similarity_array_samples

def train_pecl(model=None, freeze_resnet_fc_loaded_model=False,
               n_enc_channels=256, 
               n_layers_mlp_resnet=1, n_layers_mlp_pred=2,
               pretrained_resnet='seco', freeze_resnet=True,
               resnet_version=18, pecl_distance_metric='softmax',
               pred_train_loss='bce', use_class_weights=False,
               normalise_embedding='l2', n_bands=4,
               training_method='pecl', lr=1e-3, batch_size=64, n_epochs_max=20, 
               image_folder=None, presence_csv=None,  # None will use default paths 
               dataset_name='s2bms',
               species_process='all', p_dropout=0, temperature=0.5,
               pecl_knn=5, pecl_knn_hard_labels=False, alpha_ratio_loss=0.01,
               use_lr_scheduler=False, stop_early=False,
               verbose=1, fix_seed=42, use_mps=True,
               filepath_train_val_split=None, eval_test_set=True,
               tb_log_folder=path_dict_pecl['model_folder'],
               save_model=False, save_stats=True):
    # assert filepath_train_val_split is not None, 'Expecting filepath_train_val_split to be set.'
    assert dataset_name in ['s2bms', 'satbird-kenya', 'satbird-usawinter'], f'Dataset name {dataset_name} not implemented.'
        
    if filepath_train_val_split is None:
        if dataset_name == 's2bms':
            filepath_train_val_split = os.path.join(path_dict_pecl['repo'], 'content/split_indices_s2bms_2024-08-14-1459.pth')
        elif dataset_name == 'satbird-kenya':
            filepath_train_val_split = os.path.join(path_dict_pecl['repo'],'content/split_indices_Kenya_2024-08-14-1506.pth')
        elif dataset_name == 'satbird-usawinter':
            filepath_train_val_split = os.path.join(path_dict_pecl['repo'],'content/split_indices_USA_winter_2024-08-14-1506.pth')
    
    assert os.path.exists(filepath_train_val_split), f'File {filepath_train_val_split} does not exist.'

    if fix_seed is not None:
        pl.seed_everything(fix_seed)

    if image_folder is None:
        image_folder = path_dict_pecl[f'{dataset_name}_images']
    if presence_csv is None:
        presence_csv = path_dict_pecl[f'{dataset_name}_presence']

    stats_folder = os.path.join(tb_log_folder, 'stats')
    model_folder = os.path.join(tb_log_folder, 'full_models')
    assert os.path.exists(stats_folder), f'Folder {stats_folder} does not exist.'
    assert os.path.exists(model_folder), f'Folder {model_folder} does not exist.'

    if use_mps:
        assert torch.backends.mps.is_available()
        assert torch.backends.mps.is_built()
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=tb_log_folder)
        n_cpus = 8
        acc_use = 'gpu'
        # acc_use = 'cpu' 
    else:
        assert torch.cuda.is_available(), 'No GPU available.'
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=tb_log_folder)
        n_cpus = 8
        acc_use = 'gpu'
        
    folder_save = path_dict_pecl['model_folder']
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
        print(f'Created folder {folder_save}.')
    if verbose > 0:  # possibly also insert assert versions
        print(f'Pytorch version is {torch.__version__}') 

    ds = DataSetImagePresence(image_folder=image_folder, presence_csv=presence_csv,
                              shuffle_order_data=True, species_process=species_process,
                              augment_image=True, dataset_name=dataset_name,
                              n_bands=n_bands, zscore_im=True, mode='train')
    train_ds, val_ds, test_ds = ds.split_into_train_val(filepath=filepath_train_val_split)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus, 
                          shuffle=True, persistent_workers=True) #drop_last=True, pin_memory=True
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=n_cpus, 
                        shuffle=False,  persistent_workers=True) 
    if test_ds is not None and eval_test_set:
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=n_cpus, 
                             shuffle=False,  persistent_workers=True)
    else:
        test_dl = None

    if model is None:
        time_created = cdu.create_timestamp(include_seconds=True)
        model = ImageEncoder(n_species=ds.n_species, n_enc_channels=n_enc_channels, 
                             n_layers_mlp_resnet=n_layers_mlp_resnet, n_layers_mlp_pred=n_layers_mlp_pred,
                             pred_train_loss=pred_train_loss, 
                            pretrained_resnet=pretrained_resnet, freeze_resnet=freeze_resnet,
                            optimizer_name='Adam', resnet_version=resnet_version,
                            class_weights=ds.weights_values if use_class_weights else None,
                            pecl_distance_metric=pecl_distance_metric,
                            normalise_embedding=normalise_embedding,
                            pecl_knn=pecl_knn, pecl_knn_hard_labels=pecl_knn_hard_labels,
                            lr=lr, n_bands=n_bands, use_mps=use_mps,
                            use_lr_scheduler=use_lr_scheduler,
                            training_method=training_method,
                            alpha_ratio_loss=alpha_ratio_loss,
                            p_dropout=p_dropout, temperature=temperature,
                            time_created=time_created, batch_size_used=batch_size,
                            verbose=verbose, seed_used=fix_seed)
    else:
        assert type(model) == ImageEncoder, f'Expected model to be ImageEncoder, but got {type(model)}'
        assert model.n_species == ds.n_species, f'Number of species in model {model.n_species} does not match number of species in dataset {ds.n_species}.'
        assert model.n_enc_channels == n_enc_channels, f'Number of encoding channels in model {model.n_enc_channels} does not match number of encoding channels in dataset {n_enc_channels}.'
        assert model.n_layers_mlp_pred == n_layers_mlp_pred, f'Number of layers in prediction model {model.n_layers_mlp_pred} does not match number of layers in dataset {n_layers_mlp_pred}.'
        assert model.n_layers_mlp_resnet == n_layers_mlp_resnet, f'Number of layers in resnet model {model.n_layers_mlp_resnet} does not match number of layers in dataset {n_layers_mlp_resnet}.'
        assert model.resnet_version == resnet_version, f'Resnet version in model {model.resnet_version} does not match resnet version in dataset {resnet_version}.'
        assert model.normalise_embedding == normalise_embedding, f'Normalisation method in model {model.normalise_embedding} does not match normalisation method in dataset {normalise_embedding}.'
        assert model.n_bands == n_bands, f'Number of bands in model {model.n_bands} does not match number of bands in dataset {n_bands}.'

        model.lr = lr
        model.freeze_resnet = freeze_resnet
        model.pecl_distance_metric = pecl_distance_metric
        model.pecl_knn = pecl_knn
        model.pecl_knn_hard_labels = pecl_knn_hard_labels
        model.use_lr_scheduler = use_lr_scheduler
        model.verbose = verbose
        model.seed_used = fix_seed
        model.model_name = None  # reset model name
        model.batch_size_used = batch_size

        model.build_class_weights(class_weights=ds.weights_values if use_class_weights else None)
        model.build_training_method(training_method=training_method, pred_train_loss=pred_train_loss)
       
        if freeze_resnet_fc_loaded_model:  # freezing resnet fc head. So needs to be tuned already (hence only for loaded model).
            model.freeze_resnet_layers(freeze_all_but_last=freeze_resnet, freeze_last=True)

    cb_metrics = MetricsCallback()
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                            filename="best_checkpoint_val-{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}"),
                 cb_metrics]
    if stop_early:
        callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'))

    trainer = pl.Trainer(max_epochs=n_epochs_max, accelerator=acc_use,
                         log_every_n_steps=5,  # train loss logging steps (each step = 1 batch)
                         reload_dataloaders_every_n_epochs=1, # reload such that train_dl re-shuffles.  https://github.com/Lightning-AI/pytorch-lightning/discussions/7332
                         callbacks=callbacks, logger=tb_logger)

    timestamp_start = datetime.datetime.now()
    print(f'-- Starting training at {timestamp_start} with {n_epochs_max} epochs.')
    trainer.fit(model, train_dl, val_dl)

    timestamp_end = datetime.datetime.now()
    print(f'-- Finished training at {timestamp_end}.')


    model.store_val_metrics(metrics=cb_metrics.metrics)
    if eval_test_set:
        if test_dl is not None:
            trainer.test(dataloaders=test_dl, ckpt_path='best')
        model.store_test_metrics(metrics=cb_metrics.test_metrics)
    if save_stats:
        model.save_stats(verbose=1, folder=stats_folder)
    if save_model:
        model.save_model(verbose=1, folder=model_folder)
    return model, (train_dl, val_dl, test_dl)

def test_model(model=None, model_path=None, use_mps=True,
               filepath_train_val_split=None, 
                image_folder=None, presence_csv=None,
               fix_seed=None, species_process='all',
               save_stats=True, save_model=True,
               dataset_name='s2bms',
               tb_log_folder=path_dict_pecl['model_folder']):
    assert model is not None or model_path is not None, 'Provide either model or model_path.'
    assert not (model is not None and model_path is not None), 'Provide either model or model_path, not both.'
    stats_folder = os.path.join(tb_log_folder, 'stats')
    model_folder = os.path.join(tb_log_folder, 'full_models')
    assert os.path.exists(stats_folder), f'Folder {stats_folder} does not exist.'
    assert os.path.exists(model_folder), f'Folder {model_folder} does not exist.'

    if model_path is not None:
        model = load_model(folder=model_folder, filename=model_path, verbose=1)

    if model.test_metrics is not None:
        print('Model already has test metrics. Skipping testing.')
        return model, (None, None, None)

    if filepath_train_val_split is None:
        filepath_train_val_split = os.path.join(path_dict_pecl['repo'], 'content/split_indices_2024-03-04-1831.pth')
        assert os.path.exists(filepath_train_val_split), f'File {filepath_train_val_split} does not exist.'

    if fix_seed is not None:
        pl.seed_everything(fix_seed)
    else:
        pl.seed_everything(model.seed_used)

    if image_folder is None:
        image_folder = path_dict_pecl[f'{dataset_name}_images']
    if presence_csv is None:
        presence_csv = path_dict_pecl[f'{dataset_name}_presence']

    if use_mps:
        assert torch.backends.mps.is_available()
        assert torch.backends.mps.is_built()
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=tb_log_folder)
        n_cpus = 8
        acc_use = 'gpu'
    else:
        assert torch.cuda.is_available(), 'No GPU available.'
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=tb_log_folder)
        n_cpus = 8
        acc_use = 'gpu'

    ds = DataSetImagePresence(image_folder=image_folder, presence_csv=presence_csv,
                              shuffle_order_data=True, species_process=species_process,
                              augment_image=True, dataset_name=dataset_name,
                              n_bands=model.n_bands, zscore_im=True, mode='train')
    train_ds, val_ds, test_ds = ds.split_into_train_val(filepath=filepath_train_val_split)

    test_dl = DataLoader(test_ds, batch_size=model.batch_size_used, num_workers=n_cpus, 
                            shuffle=False,  persistent_workers=True)
    cb_metrics = MetricsCallback()
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                            filename="best_checkpoint_val-{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}"),
                 cb_metrics]
    
    trainer = pl.Trainer(max_epochs=50, accelerator=acc_use,
                         log_every_n_steps=5,  # train loss logging steps (each step = 1 batch)
                         reload_dataloaders_every_n_epochs=1, # reload such that train_dl re-shuffles.  https://github.com/Lightning-AI/pytorch-lightning/discussions/7332
                         callbacks=callbacks, logger=tb_logger)

    ckpt_folder = os.path.join(tb_log_folder, 'lightning_logs', f'version_{model.v_num}', 'checkpoints')
    contents_ckpt_folder = os.listdir(ckpt_folder)
    assert len(contents_ckpt_folder) == 1, f'Expected 1 file in folder {ckpt_folder}, but got {len(contents_ckpt_folder)}'
    assert contents_ckpt_folder[0].endswith('.ckpt'), f'File {contents_ckpt_folder[0]} should end with .ckpt'
    ckpt_path = os.path.join(ckpt_folder, contents_ckpt_folder[0])
    
    trainer.test(model, dataloaders=test_dl, ckpt_path=ckpt_path)
    model.store_test_metrics(metrics=cb_metrics.test_metrics)
    
    if save_stats:
        model.save_stats(verbose=1, folder=stats_folder)
    if save_model:
        model.save_model(verbose=1, folder=model_folder)
    return model, (None, None, test_dl)



class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback.
    https://lightning.ai/forums/t/how-to-access-the-logged-results-such-as-losses/155
    """

    def __init__(self):
        super().__init__()
        self.metrics = []
        self.test_metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.test_metrics.append(each_me)

class MeanRates(ImageEncoder):
    def __init__(self, train_ds, val_ds):
        super().__init__(training_method='pred',
                         class_weights=None,
                         pred_train_loss='bce')
        
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.calculate_mean_rates()
        self.evaluate_val_losses()

    def calculate_mean_rates(self):
        print('Calculating mean rates for training and validation datasets.')
        all_labels = []
        for sample in tqdm(self.train_ds):
            all_labels.append(sample[1][None, :])    
           
        all_labels = torch.cat(all_labels, dim=0)
        print(f'All labels shape: {all_labels.shape}')
        mean_rates = all_labels.mean(dim=0)
        self.train_mean_rates = mean_rates
        assert self.train_mean_rates.ndim == 1, self.train_mean_rates.shape
        assert self.train_mean_rates.shape[0] == self.n_species, (self.train_mean_rates.shape, self.n_species)
        assert (self.train_mean_rates >= 0).all(), self.train_mean_rates
        assert (self.train_mean_rates <= 1).all(), self.train_mean_rates

    def evaluate_val_losses(self):
        print('Evaluating validation losses for validation dataset.')
        val_loss_dict = {x: [] for x in ['mae', 'mse', 'bce', 'top_5', 'top_10', 'top_20']}
        all_val_labels = []
        for sample in tqdm(self.val_ds):
            all_val_labels.append(sample[1][None, :])
        pres_vec = torch.cat(all_val_labels, dim=0)
        len_batch = pres_vec.shape[0]

        pred_rates = self.train_mean_rates[None, :].repeat(len_batch, 1)
        assert pred_rates.shape == pres_vec.shape, (pred_rates.shape, pres_vec.shape)

        mae_loss = nn.L1Loss()(pred_rates, pres_vec)
        val_loss_dict['mae'].append(mae_loss.item())
        mse_loss = F.mse_loss(pred_rates, pres_vec)
        val_loss_dict['mse'].append(mse_loss.item())
        bce_loss = nn.BCELoss(reduction='mean')(pred_rates, pres_vec)
        val_loss_dict['bce'].append(bce_loss.item())
        for k in [5, 10, 20]:
            top_k_acc = self.top_k_accuracy(preds=pred_rates, target=pres_vec, k=k)
            val_loss_dict[f'top_{k}'].append(top_k_acc.item())

        self.val_loss_dict = val_loss_dict
        print(f'Validation losses: {val_loss_dict}')