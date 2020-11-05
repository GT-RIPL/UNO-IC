import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
from torch.utils import data
from tqdm import tqdm
from ptsemseg.models import get_model
from ptsemseg.loader import get_loaders
from ptsemseg.degredations import *
from collections import defaultdict


def validate(cfg, logdir):
    # log git commit
    import subprocess
    label = subprocess.check_output(["git", "describe", "--always"]).strip()

    # Setup seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    loaders, n_classes = get_loaders(cfg["data"]["dataset"], cfg)

    models = {}
    # Setup Model
    for model, attr in cfg["models"].items():
        attr = defaultdict(lambda: None, attr)
        models[model] = get_model(name=attr['arch'],
                                  n_classes=n_classes,
                                  input_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
                                  in_channels=attr['in_channels'],
                                  mcdo_passes=attr['mcdo_passes'],
                                  dropoutP=attr['dropoutP'],
                                  full_mcdo=attr['full_mcdo'],
                                  backbone=attr['backbone'],
                                  device=device).to(device)

        models[model] = torch.nn.DataParallel(models[model], device_ids=range(torch.cuda.device_count()))
        model_dict = models[model].state_dict()
        model_pkl = attr['resume']
        if os.path.isfile(model_pkl):
            checkpoint = torch.load(model_pkl)
            pretrained_dict = torch.load(model_pkl)['model_state']
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}
            print("Model {} parameters,Loaded {} parameters".format(len(model_dict),len(pretrained_dict)))
            model_dict.update(pretrained_dict)
            models[model].load_state_dict(pretrained_dict, strict=False)
            print("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
        else:
            print("No checkpoint found at '{}'".format(model_pkl))

    #################################################################################
    # Validation
    #################################################################################
    print("=" * 10, "Extracting", "=" * 10)
    [models[m].eval() for m in models.keys()]
    with torch.no_grad():
        length = {} 
        entropy_overall = {}                
        for m in cfg["models"].keys():
            length[m] = np.zeros(n_classes)
            entropy_overall[m] = []
        for k, valloader in loaders['val'].items():
            for i_val, (input_list, labels_list) in tqdm(enumerate(valloader)):
                images_val = {m: input_list[m][0] for m in cfg["models"].keys()}
                labels_val = labels_list[0]
                if labels_val.shape[0] <= 1:
                    continue
                # Inference
                for m in cfg["models"].keys():
                    _, entropy = models[m](images_val[m])
                    entropy_overall[m].extend(entropy.mean((1,2)).cpu().numpy().tolist())      
                    for i in range(n_classes):
                        mask = labels_val != i
                        length_temp = (labels_val == i).sum()
                        if length_temp != 0:
                            length[m][i] += length_temp

                   

        save_dir = os.path.join(logdir,'stats')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        prior = length[m]/length[m].sum()
        torch.save(prior, os.path.join(logdir,'stats','prior.pkl'))
        print(prior)
        print('saved prior at {}'.format(os.path.join(logdir,'stats','prior.pkl')))
        entropy_stats = {}
        for m in cfg["models"].keys():
            entropy_stats[m+'_mean'] = np.mean(entropy_overall[m])
            entropy_stats[m+'_std'] = np.std(entropy_overall[m])
        torch.save(entropy_stats,os.path.join(logdir,'stats','entropy.pkl'))
        print('saved entropy at {}'.format(os.path.join(logdir,'stats','entropy.pkl')))


if __name__ == "__main__":
    # python extract.py --config ./configs/synthia/eval/rgbd_synthia.yml
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/train/rgbd_BayesianSegnet_0.5_T000.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = defaultdict(lambda: None, yaml.load(fp))

    logdir = "runs" +'/'+ args.config.split("/")[2]
    if not os.path.exists(  logdir):
        os.makedirs(logdir)
    path = shutil.copy(args.config, logdir)
    validate(cfg,logdir)
    print('Done!!!')
