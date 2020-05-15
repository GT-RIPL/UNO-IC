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
import cv2

# from ptsemseg.process_img import generate_noise
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loaders
from ptsemseg.utils import get_logger, parseEightCameras, plotPrediction, plotMeansVariances, plotEntropy, plotMutualInfo, plotSpatial, save_pred, plotEverything,mutualinfo_entropy,save_stats,plotAll
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.degredations import *
from ptsemseg.fusion import *
from tensorboardX import SummaryWriter
from functools import partial
from collections import defaultdict




def validate(cfg, logdir):
    # log git commit
    import subprocess
    label = subprocess.check_output(["git", "describe", "--always"]).strip()

    # Setup seeds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    loaders, n_classes = get_loaders(cfg["data"]["dataset"], cfg)

    # Setup Metrics
    running_metrics_val = {env: runningScore(n_classes) for env in loaders['val'].keys()}
    # set seeds for training
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
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}
            print("Model {} parameters,Loaded {} parameters".format(len(model_dict),len(pretrained_dict)))
            model_dict.update(pretrained_dict)
            models[model].load_state_dict(pretrained_dict, strict=False)
            print("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
        else:
            print("No checkpoint found at '{}'".format(model_pkl))


    [models[m].eval() for m in models.keys()]
    #################################################################################
    # Validation
    #################################################################################
    print("=" * 10, "Extracting", "=" * 10)

    with torch.no_grad():
        length = np.zeros(n_classes)
        mean_stats = {}
        cov_stats = {}
        length = {} 

        for m in cfg["models"].keys():
            mean_stats[m] = [torch.zeros(n_classes,1) for _ in range(n_classes)]
            cov_stats[m] = [torch.zeros((n_classes,n_classes)) for _ in range(n_classes)]     
            length[m] = np.zeros(n_classes)
        for k, valloader in loaders['val'].items():
            for i_val, (input_list, labels_list) in tqdm(enumerate(valloader)):
                #import ipdb;ipdb.set_trace()
                inputs, labels = parseEightCameras(input_list['rgb'], labels_list, input_list['d'], device)
                inputs_display, _ = parseEightCameras(input_list['rgb_display'], labels_list, input_list['d_display'],
                                                      device)
                #import ipdb;ipdb.set_trace()
                # Read batch from only one camera
                bs = cfg['training']['batch_size']
                images_val = {m: inputs[m][:bs, :, :, :] for m in cfg["models"].keys()}
                labels_val = labels[:bs, :, :]

                if labels_val.shape[0] <= 1:
                    continue

                # Run Models
                mean = {}
                
                # Inference
                for m in cfg["models"].keys():
                    if not cfg['only_prior']:
                        mean[m], _, _ = models[m](images_val[m])     
                    for i in range(n_classes):
                        #if i != 13 and i != 14:
                        mask = labels != i
                        length_temp = (labels == i).sum()
                        if length_temp != 0:
                            length[m][i] += length_temp
                            if not cfg['only_prior']:
                                temp = mean[m].masked_fill(mask.unsqueeze(1),0).cpu()
                                mean_stats[m][i] = mean_stats[m][i] + (temp.sum((0,2,3)).unsqueeze(1)  - mean_stats[m][i]*length_temp)/length[m][i]  
                                temp_reshape = temp.transpose(1,0).reshape(n_classes,-1) #(size,batch)
                                # indices = temp_reshape.sum(1) != 0
                                    
                                cov_stats[m][i] = cov_stats[m][i] + (torch.mm(temp_reshape,temp_reshape.T) - cov_stats[m][i]*length_temp)/length[m][i]
        
        save_dir = os.path.join(logdir,'stats')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if cfg['save'] and not cfg['only_prior']:
            cov_accum = torch.zeros((n_classes,n_classes))    
            for m in cfg["models"].keys():    
                for i in range(n_classes):
                    cov_stats[m][i] = cov_stats[m][i] - torch.mm(mean_stats[m][i],mean_stats[m][i].T)
                    cov_accum +=  cov_stats[m][i] * length[m][i]
                save_dir = os.path.join(logdir,'stats',m)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                torch.save(mean_stats[m], os.path.join(save_dir,'mean.pkl'))
                torch.save(cov_stats[m], os.path.join(save_dir,'cov.pkl'))
                torch.save(cov_accum/length[m].sum(), os.path.join(save_dir,'cov_fixed.pkl'))
        log_prior = np.log(length[m]/length[m].sum())

        # stats_dir = '/'.join(logdir.split('/')[:-1])
        torch.save(log_prior, os.path.join(logdir,'stats','log_prior.pkl'))
        print(length[m]/length[m].sum())

if __name__ == "__main__":
    # python extract.py --config ./configs/synthia/rgbd_synthia.yml --save
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/train/rgbd_BayesianSegnet_0.5_T000.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--id",
        nargs="?",
        type=str,
        default=None,
        help="Unique identifier for different runs",
    )

    parser.add_argument(
        "--only_prior",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "--save",
        default=False,
        action='store_true',
    )


    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = defaultdict(lambda: None, yaml.load(fp))
    if args.id:
        cfg['id'] = args.id
    logdir = "runs" +'/'+ args.config.split("/")[2]#+'/'+cfg['id']

    # baseline train (concatenation, warping baselines)
    path = shutil.copy(args.config, logdir)

   
    # validate base model
    cfg['save'] = args.save
    cfg['only_prior'] = args.only_prior
    validate(cfg,logdir)

    print('Done!!!')
    # time.sleep(10)
