import json

from torch.utils import data

from ptsemseg.loader.synthia_loader import synthiaLoader

def get_loaders(name, cfg):
    data_loader = {
        "synthia": synthiaLoader,
    }[name]
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=cfg['data']['train_reduction'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)
        

    v_loader = {env: data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        subsplits=[env], 
        scale_quantity=cfg['data']['val_reduction'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']), ) for env in cfg['data']['val_subsplit']}

    n_classes = int(t_loader.n_classes)
    valloaders = {key: data.DataLoader(v_loader[key],
                                       batch_size=cfg['training']['batch_size'],
                                       num_workers=cfg['training']['n_workers'],) for key in v_loader.keys()}


    return {
            'train': data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True),
            'val': valloaders
        }, n_classes
