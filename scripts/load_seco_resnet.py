import os
import torch, torchvision
from pytorch_lightning.utilities.migration import pl_legacy_patch
import loadpaths_pecl
path_dict_pecl = loadpaths_pecl.loadpaths()

def load_seco_resnet(seco_ckpt_path=path_dict_pecl['seco_resnet18_1m'],
                     device_use='mps'):
    '''Load the SECO ResNet checkpoint and map weights to torch Resnet.
    
    Args:
    seco_ckpt_path: str, path to SECO checkpoint
    device_use: str, device to use (mps, gpu, cpu, etc.)
    
    Returns:
    seco_sd: dict, state dict of SECO checkpoint
    '''
    
    assert os.path.exists(seco_ckpt_path)

    with pl_legacy_patch():
        seco_sd = torch.load(seco_ckpt_path, 
                             map_location=device_use)['state_dict']
    return seco_sd

def load_resnet(resnet_name='resnet50'):
    '''Load a torchvision ResNet model.'''
    if resnet_name == 'resnet50':
        resnet = torchvision.models.resnet50(pretrained=False)
    elif resnet_name == 'resnet18':
        resnet = torchvision.models.resnet18(pretrained=False)
    else:
        raise ValueError('Only resnet50 and resnet18 are supported')
    return resnet

def map_seco_to_torchvision_weights(model=None, prefix_resnet_layernames='',
                                    seco_ckpt_path=path_dict_pecl['seco_resnet18_1m'],
                                    device_use='mps', resnet_name='resnet18',
                                    encoder_use='q', verbose=1):
    '''Map SECO weights to torchvision weights. All layers are mapped 
    except for the head (fully connected layers for regular Resnet, or
    decoder for UNet).

    Args:
    model: torch.nn.Module, model to map weights to. If None, load a ResNet model.
    prefix_resnet_layernames: str, prefix of layer names in model (eg 'base.encoder.' 
                                   for smp.UNet. Use '' for torchvision ResNet.)
    seco_ckpt_path: str, path to SECO checkpoint
    device_use: str, device to use (mps, gpu, cpu, etc.)
    encoder_use: str, 'q' or 'k' (see SECO paper)
    verbose: int, verbosity level

    Returns:
    model: torch.nn.Module, model with SECO weights
    '''
    assert encoder_use in ['q', 'k'], 'See SECO paper; must be q or k'
    
    seco_sd = load_seco_resnet(seco_ckpt_path=seco_ckpt_path, device_use=device_use)
    if model is None:
        model = load_resnet(resnet_name=resnet_name)
        prefix_resnet_layernames = ''  # no prefix for torchvision ResNet

    ## If mapping to a smp.UNet, use:
    # prefix_resnet_layernames = 'base.encoder.'
    
    seco_sd_keys = list(seco_sd.keys())
    resnet_keys = list(model.state_dict().keys())

    if verbose:
        print('seco keys:', seco_sd_keys)
        print('resnet keys:', resnet_keys)

    mapping_seco_to_original = {}
    encoder_not_use = 'k' if encoder_use == 'q' else 'q'
    map_layer_name = {'4': 'layer1', '5': 'layer2', 
                        '6': 'layer3', '7': 'layer4'}  # remap SECO layer names to torchvision layer names
    
    for k in seco_sd_keys:  # loop over SECO layers
        if k[:9] == f'encoder_{encoder_use}':  # keep
            new_k = k.replace(f'encoder_{encoder_use}.', prefix_resnet_layernames)
        elif k[:9] == f'encoder_{encoder_not_use}':  # discard
            continue
        elif k[:5] == 'queue':  # discard
            continue 
        elif k[:7] == f'heads_{encoder_use}':  # discard
            continue 
        elif k[:7] == f'heads_{encoder_not_use}':  # discard
            continue 
        else:  # flag unexpected outcomes
            raise ValueError(k)
        
        if len(k.split('.')) > 3:  # most layers
            assert k[10] in map_layer_name.keys(), k[10]  # k is something like "encoder_q.4.0.bn1.weight", so k[10] is the layer number
            new_k = new_k.replace(f'{k[10]}.', f'{map_layer_name[k[10]]}.')
        elif len(k.split('.')) == 3:  # frew special cases
            if k[10] == '0':
                new_k = new_k.replace('0.', 'conv1.')
            elif k[10] == '1':
                new_k = new_k.replace('1.', 'bn1.')
            else:
                raise ValueError(k)
            
        mapping_seco_to_original[k] = new_k
        
    # mapping_seco_to_original[f'heads_{encoder_use}.0.0.weight'] = 'base.segmentation_head.0.weight'
    # mapping_seco_to_original[f'heads_{encoder_use}.0.0.bias'] = 'base.segmentation_head.0.bias'

    ## create new state dict:
    new_sd = {}
    for k, v in seco_sd.items():
        if k in mapping_seco_to_original.keys():
            new_sd[mapping_seco_to_original[k]] = v
        else:  # many layers that we won't use (eg of other encoder), so suppress warnings
            # print(f'Key {k} not found in original state dict')
            pass

    print(f'Original state dict had {len(resnet_keys)} keys. Recovered {len(new_sd.keys())} keys from SECO checkpoint.')
    # print(f'Missing keys: {set(original_sd_keys) - set(new_sd.keys())}')
    set_missing_keys = set(resnet_keys) - set(new_sd.keys())
    if verbose > 0:
        print(f'Missing keys: {set_missing_keys}')
    for k in set_missing_keys:
        if model is None:
            assert k[:3] == 'fc.', k  # only the fully connected layers are missing. Possibly use heads_q.0.0.weight and heads_q.0.0.bias ? 
        ## If mapping to a smp.UNet, use:
        # assert k[:12] in ['base.decoder', 'base.segment'], k
        else:
            if verbose > 0:
                print(f'Key {k} not found in SECO state dict')
    print('No unexpected missing keys (only decoder layers are missing).')

    ## load new state dict:
    model.load_state_dict(new_sd, strict=False)  # strict=False because we don't want to load the decoder layers
    return model