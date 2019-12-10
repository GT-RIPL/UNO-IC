import json

from torch.utils import data
from ptsemseg.augmentations import get_composed_augmentations

from ptsemseg.loader.airsim_loader import airsimLoader
from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import (
    MITSceneParsingBenchmarkLoader
)
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader
from ptsemseg.loader.synthia_loader import synthiaLoader



def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "airsim": airsimLoader,
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "synthia": synthiaLoader
    }[name]

def get_loaders(name, cfg):

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)
    
    data_loader = {
        "airsim": airsimLoader,
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "synthia": synthiaLoader
    }[name]
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=cfg['data']['train_reduction'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    # r_loader = data_loader(
    #     data_path,
    #     is_transform=True,
    #     split=cfg['data']['recal_split'],
    #     subsplits=cfg['data']['recal_subsplit'],
    #     scale_quantity=cfg['data']['recal_reduction'],
    #     img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
    #     augmentations=data_aug)

    tv_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=0.05,
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = {env: data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'], subsplits=[env], scale_quantity=cfg['data']['val_reduction'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']), ) for env in cfg['data']['val_subsplit']}

    n_classes = int(t_loader.n_classes)

    valloaders = {key: data.DataLoader(v_loader[key],
                                       batch_size=cfg['training']['batch_size'],
                                       num_workers=cfg['training']['n_workers']) for key in v_loader.keys()}

    # add training samples to validation sweep
    # valloaders = {**valloaders, 'train': data.DataLoader(tv_loader,
                                                         # batch_size=cfg['training']['batch_size'],
                                                         # num_workers=cfg['training']['n_workers'])}

    return {
            'train': data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True),
            # 'recal': data.DataLoader(r_loader,
            #                       batch_size=cfg['training']['batch_size'],
            #                       num_workers=cfg['training']['n_workers'],
            #                       shuffle=True),
            'val': valloaders
        }, n_classes


def get_data_path(name, config_file="config.json"):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]["data_path"]
