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
from ptsemseg.models.fusion.fusion import *
from tensorboardX import SummaryWriter
from functools import partial
from collections import defaultdict


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def validate(cfg, writer, logger, logdir):
    # log git commit
    import subprocess
    label = subprocess.check_output(["git", "describe", "--always"]).strip()
    logger.info("Using commit {}".format(label))

    # Setup seeds
    random_seed(1337, True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    loaders, n_classes = get_loaders(cfg["data"]["dataset"], cfg)

    # Setup Metrics
    running_metrics_val = {env: runningScore(n_classes) for env in loaders['val'].keys()}
    # Setup Meters
    val_loss_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}

    scale_logits = GlobalScaling(cfg['training_stats'])

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

        # Load pretrained weights
        # if str(attr['resume']) != "None" or str(attr['resume_temp']) != "None" :
        model_dict = models[model].state_dict()
        
        # model_pkl_dict = {}

        # if attr['resume'] != "None":
        #     model_pkl_dict["single"]= attr['resume']
        #     #if attr['resume'] == 'same_yaml':
        #     #    model_pkl = "{}/{}_pspnet_airsim_best_model.pkl".format(logdir, model)

        # if attr['resume_temp'] != "None":
        #     model_pkl_dict["temp"] = attr['resume_temp']

        # if attr['resume_temp_d'] != "None":
        #     model_pkl_dict["temp_d"] = attr['resume_temp_d']

        # if attr['resume_temp_rgb'] != "None":
        #     model_pkl_dict["temp_rgb"] = attr['resume_temp_rgb']
        model_pkl = attr['resume']
        # for model_key,model_pkl in model_pkl_dict.items():
        if os.path.isfile(model_pkl):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(model_pkl)
            )
            checkpoint = torch.load(model_pkl)
            pretrained_dict = torch.load(model_pkl)['model_state']
            # pretrained_dict = {}

            # if model_key == "temp":
            #     for wieghts_key,weights in pretrained_dict_temp.items():
            #         if wieghts_key.split('.')[1]=='segnet':
            #             pretrained_dict[wieghts_key] = weights
            #         else:
            #             pretrained_dict['module.tempnet'+wieghts_key.split('module')[1]] = weights
            # elif model_key == "comp":
            #     for wieghts_key,weights in pretrained_dict_temp.items():
            #         if wieghts_key.split('.')[1]=='segnet':
            #             pretrained_dict[wieghts_key] = weights
            #         else:
            #             pretrained_dict['module.compnet'+wieghts_key.split('module')[1]] = weights 
            # elif model_key == "temp_d":
            #     for wieghts_key,weights in pretrained_dict_temp.items():
            #         if wieghts_key.split('.')[1]!='segnet':
            #             #pretrained_dict[wieghts_key] = weights
            #         #else:
            #             pretrained_dict['module.tempnet_d'+wieghts_key.split('module')[1]] = weights

            # elif model_key == "temp_rgb":
            #     for wieghts_key,weights in pretrained_dict_temp.items():
            #         if wieghts_key.split('.')[1]!='segnet':
            #             #pretrained_dict[wieghts_key] = weights
            #         #else:
            #             pretrained_dict['module.tempnet_rgb'+wieghts_key.split('module')[1]] = weights
            # else:
            # pretrained_dict = pretrained_dict_temp
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}
            print("Model {} parameters,Loaded {} parameters".format(len(model_dict),len(pretrained_dict)))
            #import ipdb;ipdb.set_trace()
            model_dict.update(pretrained_dict)
            models[model].load_state_dict(pretrained_dict, strict=False)
            logger.info("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
            print("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
        else:
            logger.info("No checkpoint found at '{}'".format(model_pkl))
            print("No checkpoint found at '{}'".format(model_pkl))


    [models[m].eval() for m in models.keys()]
    #################################################################################
    # Validation
    #################################################################################
    print("=" * 10, "VALIDATING", "=" * 10)

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
                    mean[m], _, _ = models[m](images_val[m])     
                    for i in range(n_classes):
                        mask = labels != i
                        length[m][i] += mask.sum()
                        temp = mean[m].masked_fill(mask.unsqueeze(1),0).cpu()
                        mean_stats[m][i] = mean_stats[m][i] + (temp.sum((0,2,3)).unsqueeze(1)  - mean_stats[m][i])/length[m][i]  
                        temp_reshape = temp.transpose(0,1).reshape(n_classes,-1)
                        # indices = temp_reshape.sum(1) != 0
                            
                        cov_stats[m][i] = cov_stats[m][i] + (torch.mm(temp_reshape,temp_reshape.T) - cov_stats[m][i])/length[m][i]
                        import ipdb;ipdb.set_trace()



        for m in cfg["models"].keys():    
            for i in range(n_classes):
                cov_stats[m][i] = cov_stats[m][i] - torch.mm(mean_stats[m][i],mean_stats[m][i].T)
            save_dir = os.path.join(logdir,'stats',m)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(cov_stats[m], os.path.join(save_dir,'cov.pkl'))
            torch.save(mean_stats[m], os.path.join(save_dir,'mean.pkl'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/train/rgbd_BayesianSegnet_0.5_T000.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--tag",
        nargs="?",
        type=str,
        default="",
        help="Unique identifier for different runs",
    )

    parser.add_argument(
        "--run",
        nargs="?",
        type=str,
        default="",
        help="Directory to rerun",
    )

    args = parser.parse_args()

    # cfg is a  with two-level dictionary ['training','data','model']['batch_size']
    if args.run != "":

        # find and load config
        for root, dirs, files in os.walk(args.run):
            for f in files:
                if '.yml' in f:
                    path = root + f
                    args.config = path

        with open(path) as fp:
            cfg = defaultdict(lambda: None, yaml.load(fp))

        # find and load saved best models
        for m in cfg['models'].keys():
            for root, dirs, files in os.walk(args.run):
                for f in files:
                    if m in f and '.pkl' in f:
                        cfg['models'][m]['resume'] = root + f

        logdir = args.run

    else:
        with open(args.config) as fp:
            cfg = defaultdict(lambda: None, yaml.load(fp))

        logdir = "/".join(["runs"] + args.config.split("/")[1:])[:-4]+'/'+cfg['id']

        # append tag
        if args.tag:
            logdir += "/" + args.tag

    # baseline train (concatenation, warping baselines)
    writer = SummaryWriter(logdir)
    path = shutil.copy(args.config, logdir)
    logger = get_logger(logdir)

    # generate seed if none present
    if cfg['seed'] is None:
        seed = int(time.time())
        cfg['seed'] = seed

        # modify file to reflect seed
        with open(path, 'r') as original:
            data = original.read()
        with open(path, 'w') as modified:
            modified.write("seed: {}\n".format(seed) + data)

    # validate base model
    validate(cfg, writer, logger, logdir)

    print('Done!!!')
    time.sleep(10)
    writer.close()
