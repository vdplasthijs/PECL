import sys, os
import pytest
import torch 
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from DataSetImagePresence import DataSetImagePresence
import paired_embeddings_models as pem
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()

@pytest.fixture
def create_model(create_ds, request):
    use_mock = request.config.getoption("--use-mock")
    model = pem.ImageEncoder(n_species=create_ds.n_species, n_enc_channels=256, 
                             n_layers_mlp_resnet=1, n_layers_mlp_pred=2,
                             pred_train_loss='bce', 
                            pretrained_resnet='seco', freeze_resnet=True,
                            optimizer_name='Adam', resnet_version=18,
                            class_weights=None, # create_ds.weights_values if use_class_weights else None,
                            pecl_distance_metric='softmax',
                            normalise_embedding='l2',
                            pecl_knn=2, pecl_knn_hard_labels=False,
                            lr=1e-3, n_bands=create_ds.n_bands, use_mps=True,
                            use_lr_scheduler=False,
                            training_method='pecl',
                            alpha_ratio_loss=0.5,
                            p_dropout=0, temperature=0.5,
                            time_created=None, 
                            batch_size_used=4 if use_mock else 64,
                            verbose=1, seed_used=0)
    return model

@pytest.fixture
def create_split_dls(create_split_ds, create_model):
    '''In test_model.py because model's batch_size is used.'''
    bs = create_model.batch_size_used
    train_ds, val_ds, test_ds = create_split_ds
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False)
    return (train_dl, val_dl, test_dl)  

@pytest.mark.fast
def test_default_correspondence_ds(create_ds, create_model):
    assert create_ds.n_bands == create_model.n_bands
    assert create_ds.n_species == create_model.n_species

@pytest.mark.fast
def test_default_settings_model(create_model):
    assert create_model.normalise_embedding == 'l2', 'Currently expecting l2 normalisation'
    assert hasattr(create_model, 'forward_pass')
    assert hasattr(create_model, 'pred_train_loss')
    assert hasattr(create_model, 'resnet')
    assert hasattr(create_model, 'prediction_model')
    assert hasattr(create_model, 'forward')
    assert hasattr(create_model, 'save_stats')
    assert hasattr(create_model, 'save_model')

@pytest.mark.slow
def test_forward_pass(create_model, create_split_dls):
    train_dl, val_dl, test_dl = create_split_dls
    for batch in train_dl:
        assert len(batch) == 2
        assert batch[0].shape == (create_model.batch_size_used, create_model.n_bands, 224, 224)
        assert batch[1].shape == (create_model.batch_size_used, create_model.n_species)
        break 

    create_model.pred_train_loss = 'bce'  # to avoid error (because create_model initialises with training_method='pecl' which sets pred_train_loss to None)
    for pass_name in ['forward_pass', 'pecl_pass', 'pred_and_pecl_pass', 'pred_pass']:
        loss, _, im_enc = getattr(create_model, pass_name)(batch)
        loss = loss.detach()
        assert type(loss) == torch.Tensor
        assert type(im_enc) == torch.Tensor 
        assert im_enc.shape == (create_model.batch_size_used, create_model.n_enc_channels)

    for step_name in ['training_step', 'validation_step', 'test_step']:
        loss = getattr(create_model, step_name)(batch, 0)
        loss = loss.detach()
        assert type(loss) == torch.Tensor

@pytest.mark.slow
def test_pass_using_mocked_data(create_model):
    '''Test forward pass using mocked data.'''
    bs = create_model.batch_size_used
    n_bands = create_model.n_bands
    n_species = create_model.n_species
    batch = (torch.randn(bs, n_bands, 224, 224), torch.rand(bs, n_species))
    create_model.pred_train_loss = 'bce'  # to avoid error (because create_model initialises with training_method='pecl' which sets pred_train_loss to None)
    for pass_name in ['forward_pass', 'pecl_pass', 'pred_and_pecl_pass', 'pred_pass']:
        loss, _, im_enc = getattr(create_model, pass_name)(batch)
        loss = loss.detach()
        assert type(loss) == torch.Tensor
        assert type(im_enc) == torch.Tensor 
        assert im_enc.shape == (bs, create_model.n_enc_channels)
