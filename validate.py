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
from ptsemseg.utils import get_logger, parseEightCameras, plotPrediction, plotMeansVariances, plotEntropy, \
    plotMutualInfo
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.degredations import *

from tensorboardX import SummaryWriter
from scipy.misc import imsave

from functools import partial
from collections import defaultdict


def validate(cfg, writer, logger, logdir):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    loaders, n_classes = get_loaders(cfg["data"]["dataset"], cfg)

    # Setup Metrics
    running_metrics_val = {env: runningScore(n_classes) for env in loaders['val'].keys()}

    # Setup Meters
    val_loss_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}
    val_CE_loss_meter = {env: averageMeter() for env in loaders['val'].keys()}
    val_REG_loss_meter = {env: averageMeter() for env in loaders['val'].keys()}
    variance_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}
    entropy_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}
    mutual_info_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}
    time_meter = averageMeter()

    start_iter = 0
    models = {}
    swag_models = {}
    optimizers = {}
    schedulers = {}

    # Setup Model
    for model, attr in cfg["models"].items():

        attr = defaultdict(lambda: None, attr)

        models[model] = get_model(cfg["model"],
                                  n_classes,
                                  input_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
                                  batch_size=cfg["training"]["batch_size"],
                                  in_channels=attr['in_channels'],
                                  start_layer=attr['start_layer'],
                                  end_layer=attr['end_layer'],
                                  mcdo_passes=attr['mcdo_passes'],
                                  dropoutP=attr['dropoutP'],
                                  full_mcdo=cfg['full_mcdo'],
                                  reduction=attr['reduction'],
                                  device=device,
                                  recalibrator=cfg['recalibrator'],
                                  temperatureScaling=cfg['temperatureScaling'],
                                  varianceScaling=cfg['varianceScaling'],
                                  freeze=attr['freeze'],
                                  bins=cfg["bins"]).to(device)

        models[model] = torch.nn.DataParallel(models[model], device_ids=range(torch.cuda.device_count()))

        # Setup optimizer, lr_scheduler and loss function
        optimizer_cls = get_optimizer(cfg)
        optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                            if k != 'name'}

        optimizers[model] = optimizer_cls(models[model].parameters(), **optimizer_params)
        logger.info("Using optimizer {}".format(optimizers[model]))

        schedulers[model] = get_scheduler(optimizers[model], cfg['training']['lr_schedule'])

        loss_fn = get_loss_function(cfg)
        logger.info("Using loss {}".format(loss_fn))

        # Load pretrained weights
        if str(attr['resume']) is not "None":

            model_pkl = attr['resume']
            if attr['resume'] == 'same_yaml':
                model_pkl = "{}/{}_pspnet_airsim_best_model.pkl".format(logdir, model)

            if os.path.isfile(model_pkl):
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(model_pkl)
                )
                checkpoint = torch.load(model_pkl)

                pretrained_dict = torch.load(model_pkl)['model_state']
                model_dict = models[model].state_dict()

                # 1. filter out unnecessary keys
                pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                        k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}

                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)

                # 3. load the new state dict
                models[model].load_state_dict(pretrained_dict, strict=False)

                if attr['resume'] == 'same_yaml':
                    optimizers[model].load_state_dict(checkpoint["optimizer_state"])
                    schedulers[model].load_state_dict(checkpoint["scheduler_state"])
                    start_iter = checkpoint["epoch"]
                else:
                    start_iter = checkpoint["epoch"]

                # start_iter = 0
                logger.info("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
            else:
                logger.info("No checkpoint found at '{}'".format(model_pkl))

    best_iou = -100.0
    i = start_iter
    print(i)

    [models[m].eval() for m in models.keys()]

    #################################################################################
    # Recalibration
    #################################################################################
    if cfg["recal"] != "None":
        print("=" * 10, "RECALIBRATING", "=" * 10)

        for m in cfg["models"].keys():

            bs = cfg['training']['batch_size']
            output_all = torch.zeros(
                (len(loaders['recal']) * bs, n_classes, cfg['data']['img_rows'], cfg['data']['img_cols']))
            labels_all = torch.zeros(
                (len(loaders['recal']) * bs, cfg['data']['img_rows'], cfg['data']['img_cols']), dtype=torch.long)

            with torch.no_grad():
                for i_recal, (images_list, labels_list, aux_list) in tqdm(enumerate(loaders['recal'])):
                    inputs, labels = parseEightCameras(images_list, labels_list, aux_list, device)

                    # Read batch from only one camera
                    images_recal = inputs[m][:bs, :, :, :]
                    labels_recal = labels[:bs, :, :]

                    # Run Models
                    mean, variance = models[m].module.forwardMCDO(images_recal)

                    # concat results
                    output_all[bs * i_recal:bs * (i_recal + 1), :, :, :] = torch.nn.Softmax(dim=1)(mean)
                    labels_all[bs * i_recal:bs * (i_recal + 1), :, :] = labels_recal

            # fit calibration models
            for c in range(n_classes):
                models[m].module.calibrationPerClass[c].fit(output_all, labels_all)
            models[m].module.showCalibration(output_all, labels_all, logdir, m, i)

    #################################################################################
    # Validation
    #################################################################################
    print("=" * 10, "VALIDATING", "=" * 10)

    with torch.no_grad():
        for k, valloader in loaders['val'].items():
            for i_val, (input_list, labels_list) in tqdm(enumerate(valloader)):

                inputs, labels = parseEightCameras(input_list['rgb'], labels_list, input_list['d'], device)
                inputs_display, _ = parseEightCameras(input_list['rgb_display'], labels_list, input_list['d_display'],
                                                      device)

                # Read batch from only one camera
                bs = cfg['training']['batch_size']
                images_val = {m: inputs[m][:bs, :, :, :] for m in cfg["models"].keys()}
                labels_val = labels[:bs, :, :]

                if labels_val.shape[0] <= 1:
                    continue

                # Run Models
                mean = {}
                variance = {}
                entropy = {}
                mutual_info = {}
                val_loss = {}

                for m in cfg["models"].keys():


                    if cfg['swa']:
                        mean[m] = swag_models[m](images_val[m])
                        variance[m] = torch.zeros(mean[m].shape)
                        entropy[m] = torch.zeros(labels_val.shape)
                        mutual_info[m] = torch.zeros(labels_val.shape)
                    elif hasattr(models[m].module, 'forwardMCDO'):
                        mean[m], variance[m], entropy[m], mutual_info[m] = models[m].module.forwardMCDO(
                            images_val[m], recalType=cfg["recal"])
                    else:
                        mean[m] = models[m](images_val[m])
                        variance[m] = torch.zeros(mean[m].shape)
                        entropy[m] = torch.zeros(labels_val.shape)
                        mutual_info[m] = torch.zeros(labels_val.shape)
                    val_loss[m] = loss_fn(input=mean[m], target=labels_val)

                # import ipdb;ipdb.set_trace()
                # Fusion Type
                if cfg["fusion"] == "None":
                    outputs = torch.nn.Softmax(dim=1)(mean[list(cfg["models"].keys())[0]])
                elif cfg["fusion"] == "SoftmaxMultiply":
                    outputs = torch.nn.Softmax(dim=1)(mean["rgb"]) * torch.nn.Softmax(dim=1)(mean["d"])
                elif cfg["fusion"] == "SoftmaxAverage":
                    outputs = torch.nn.Softmax(dim=1)(mean["rgb"]) + torch.nn.Softmax(dim=1)(mean["d"])
                elif cfg["fusion"] == "WeightedVariance":
                    rgb_var = 1 / (torch.mean(variance["rgb"], 1) + 1e-5)
                    d_var = 1 / (torch.mean(variance["d"], 1) + 1e-5)

                    rgb = torch.nn.Softmax(dim=1)(mean["rgb"])
                    d = torch.nn.Softmax(dim=1)(mean["d"])
                    for n in range(n_classes):
                        rgb[:, n, :, :] = rgb[:, n, :, :] * rgb_var
                        d[:, n, :, :] = d[:, n, :, :] * d_var
                    outputs = rgb + d

                elif cfg["fusion"] == "FuzzyLogic":
                    outputs = torch.max(torch.nn.Softmax(dim=1)(mean["rgb"]),
                                        torch.nn.Softmax(dim=1)(mean["d"]))
                elif cfg["fusion"] == "FuzzyLogicVariance":
                    outputs = torch.max(variance["rgb"], variance["d"])
                else:
                    print("Fusion Type Not Supported")

                # plot ground truth vs mean/variance of outputs
                pred = outputs.argmax(1).cpu().numpy()
                gt = labels_val.data.cpu().numpy()
                # import ipdb;ipdb.set_trace()
                if i_val % cfg["training"]["png_frames"] == 0:
                    plotPrediction(logdir, cfg, n_classes, i + 1, i_val, k, inputs_display, pred, gt)
                    for m in cfg["models"].keys():
                        plotMeansVariances(logdir, cfg, n_classes, i + 1, i_val, m, k + "/" + m, inputs_display,
                                           pred, gt, mean[m], variance[m])
                        plotEntropy(logdir, i + 1, i_val, k + "/" + m, pred, entropy[m])
                        plotMutualInfo(logdir, i + 1, i_val, k + "/" + m, pred, mutual_info[m])

                running_metrics_val[k].update(gt, pred)

        for m in cfg["models"].keys():
            for k in loaders['val'].keys():
                writer.add_scalar('loss/val_loss/{}/{}'.format(m, k), val_loss_meter[m][k].avg, i + 1)
                logger.info("%s %s Iter %d Loss: %.4f" % (m, k, i + 1, val_loss_meter[m][k].avg))

    for env, valloader in loaders['val'].items():
        score, class_iou = running_metrics_val[env].get_scores()
        for k, v in score.items():
            logger.info('{}: {}'.format(k, v))
            writer.add_scalar('val_metrics/{}/{}'.format(env, k), v, i + 1)

        for k, v in class_iou.items():
            logger.info('{}: {}'.format(k, v))
            writer.add_scalar('val_metrics/{}/cls_{}'.format(env, k), v, i + 1)

        for m in cfg["models"].keys():
            val_loss_meter[m][env].reset()
        running_metrics_val[env].reset()
    


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
    if len(args.run) > 0:

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

        logdir = "/".join(["runs"] + args.config.split("/")[1:])[:-4]

        shutil.copy(args.config, logdir)
        
    # generate seed if none present
    if cfg['seed'] is None:
        seed = int(time.time())
        cfg['seed'] = seed

        # modify file to reflect seed
        with open(path, 'r') as original:
            data = original.read()
        with open(path, 'w') as modified:
            modified.write("seed: {}\n".format(seed) + data)

    # append tag 
    if args.tag:
        logdir += "/" + args.tag
    writer = SummaryWriter(logdir)

    print("RUNDIR: {}".format(logdir))
    logger = get_logger(logdir)

    # validate base model
    validate(cfg, writer, logger, logdir)
    
    print('done')
    time.sleep(10)
    writer.close()
