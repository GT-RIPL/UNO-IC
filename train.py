import matplotlib

matplotlib.use('Agg')

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
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter
from scipy.misc import imsave

from functools import partial

import matplotlib.pyplot as plt
from collections import defaultdict

def train(cfg, writer, logger, logdir):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=cfg['data']['train_reduction'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    r_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['recal_split'],
        subsplits=cfg['data']['recal_subsplit'],
        scale_quantity=cfg['data']['recal_reduction'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

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
        split="val", subsplits=[env], scale_quantity=cfg['data']['val_reduction'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']), ) for env in cfg['data']['val_subsplit']}

    n_classes = int(t_loader.n_classes)
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True)

    recalloader = data.DataLoader(r_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True)

    valloaders = {key: data.DataLoader(v_loader[key],
                                       batch_size=cfg['training']['batch_size'],
                                       num_workers=cfg['training']['n_workers']) for key in v_loader.keys()}

    # add training samples to validation sweep
    valloaders = {**valloaders, 'train': data.DataLoader(tv_loader,
                                                         batch_size=cfg['training']['batch_size'],
                                                         num_workers=cfg['training']['n_workers'])}

    # Setup Metrics
    running_metrics_val = {env: runningScore(n_classes) for env in valloaders.keys()}

    # Setup Meters
    val_loss_meter = {m: {env: averageMeter() for env in valloaders.keys()} for m in cfg["models"].keys()}
    val_CE_loss_meter = {env: averageMeter() for env in valloaders.keys()}
    val_REG_loss_meter = {env: averageMeter() for env in valloaders.keys()}
    time_meter = averageMeter()

    start_iter = 0
    models = {}
    optimizers = {}
    schedulers = {}

    # Setup Model
    for model, attr in cfg["models"].items():

        if 'full_mcdo' in cfg.keys():
            full_mcdo = cfg['full_mcdo']
        else:
            full_mcdo = False

        models[model] = get_model(cfg["model"],
                                  n_classes,
                                  input_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
                                  batch_size=cfg["training"]["batch_size"],
                                  in_channels=attr['in_channels'],
                                  start_layer=attr['start_layer'],
                                  end_layer=attr['end_layer'],
                                  mcdo_passes=attr['mcdo_passes'],
                                  dropoutP=attr['dropoutP'],
                                  full_mcdo=full_mcdo,
                                  reduction=attr['reduction'],
                                  device=device,
                                  recalibrator=cfg['recalibrator'],
                                  temperatureScaling=cfg['temperatureScaling'],
                                  freeze=attr['freeze'],
                                  bins=cfg["bins"]).to(device)

        if "caffemodel" in attr['resume']:
            models[model].load_pretrained_model(model_path=attr['resume'])

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
        if str(attr['resume']) is not "None" and not "caffemodel" in attr['resume']:

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


    while i <= cfg["training"]["train_iters"]:

        print("=" * 10, "TRAINING", "=" * 10)
        for (images_list, labels_list, aux_list) in trainloader:

            i += 1
            #################################################################################
            # Training
            #################################################################################
            if i < cfg["training"]["train_iters"]:
                inputs, labels = parseEightCameras(images_list, labels_list, aux_list, device)

                # Read batch from only one camera
                bs = cfg['training']['batch_size']
                images = {m: inputs[m][:bs, :, :, :] for m in cfg["models"].keys()}
                labels = labels[:bs, :, :]

                if labels.shape[0] <= 1:
                    continue

                start_ts = time.time()

                [schedulers[m].step() for m in schedulers.keys()]
                [models[m].train() for m in models.keys()]
                [optimizers[m].zero_grad() for m in optimizers.keys()]

                # Run Models
                outputs = {}
                loss = {}
                for m in cfg["models"].keys():
                    outputs[m] = models[m](images[m])

                    loss[m] = loss_fn(input=outputs[m], target=labels)
                    loss[m].backward()
                    optimizers[m].step()

                time_meter.update(time.time() - start_ts)
                if (i + 1) % cfg['training']['print_interval'] == 0:
                    for m in cfg["models"].keys():
                        fmt_str = "Iter [{:d}/{:d}]  Loss {}: {:.4f}  Time/Image: {:.4f}"
                        print_str = fmt_str.format(i + 1,
                                                   cfg['training']['train_iters'],
                                                   m,
                                                   loss[m].item(),
                                                   time_meter.avg / cfg['training']['batch_size'])

                        print(print_str)
                        logger.info(print_str)
                        writer.add_scalar('loss/train_loss/' + m, loss[m].item(), i + 1)
                    time_meter.reset()

            if i % cfg["training"]["val_interval"] == 0 or i >= cfg["training"]["train_iters"]:

                [models[m].eval() for m in models.keys()]

                #################################################################################
                # Recalibration
                #################################################################################
                if cfg["recal"] != "None":
                    print("=" * 10, "RECALIBRATING", "=" * 10)

                    for m in cfg["models"].keys():

                        bs = cfg['training']['batch_size']
                        output_all = torch.zeros(
                            (len(recalloader) * bs, n_classes, cfg['data']['img_rows'], cfg['data']['img_cols']))
                        labels_all = torch.zeros(
                            (len(recalloader) * bs, cfg['data']['img_rows'], cfg['data']['img_cols']), dtype=torch.long)

                        with torch.no_grad():
                            for i_recal, (images_list, labels_list, aux_list) in tqdm(enumerate(recalloader)):
                                inputs, labels = parseEightCameras(images_list, labels_list, aux_list, device)

                                # Read batch from only one camera
                                images_recal = inputs[m][:bs, :, :, :]
                                labels_recal = labels[:bs, :, :]

                                # Run Models
                                mean, variance = models[m].module.forwardMCDO(images_recal)

                                # concat results
                                output_all[bs * i_recal:bs * (i_recal + 1), :, :, :] = mean
                                labels_all[bs * i_recal:bs * (i_recal + 1), :, :] = labels_recal

                        # fit calibration models
                        for c in range(n_classes):
                            models[m].module.calibrationPerClass[c].fit(output_all, labels_all)
                        models[m].module.showCalibration(output_all, labels_all, logdir, m, i)

                    # plot mean/variances of predictions of (un)calibrated models
                    with torch.no_grad():
                        for i_recal, (images_list, labels_list, aux_list) in tqdm(enumerate(recalloader)):

                            inputs, labels = parseEightCameras(images_list, labels_list, aux_list, device)

                            # Read batch from only one camera
                            bs = cfg['training']['batch_size']
                            images_recal = {m: inputs[m][:bs, :, :, :] for m in cfg["models"].keys()}
                            labels_recal = labels[:bs, :, :]
                            gt = labels_recal.data.cpu().numpy()

                            # Run Models
                            for m in cfg["models"].keys():
                                outputs, mean, variance = models[m](images_recal[m])

                                # plot predictions without calibration
                                pred = mean.data.max(1)[1].cpu().numpy()
                                plotMeansVariances(logdir, cfg, n_classes, i, i_recal, m, "recal/pre_recal", inputs,
                                                   pred, gt, mean, variance)
                                plotPrediction(logdir, cfg, n_classes, i, i_recal, "recal/" + m + "/pre_recal_pred",
                                               inputs, pred, gt)

                                # plot predictions with calibration
                                # TODO calibrate outputs instead of rereunning model with calibration
                                # outputs = models[m].module.calibrateOutput(output)
                                if hasattr(models[m].module, 'forwardMCDO'):
                                    mean[m], variance[m] = models[m].module.forwardMCDO(images_val[m], cfg["recal"])
                                else:
                                    mean[m] = models[m](images_val[m])
                                    variance[m] = torch.zeros(mean[m].shape)
                                post_pred = mean.data.max(1)[1].cpu().numpy()
                                plotMeansVariances(logdir, cfg, n_classes, i, i_recal, m, "recal/post_recal", inputs,
                                                   post_pred, gt, mean, variance)
                                plotPrediction(logdir, cfg, n_classes, i, i_recal, "recal/" + m + "/post_recal_pred",
                                               inputs, post_pred, gt)

                                torch.cuda.empty_cache()
                    
                #################################################################################
                # Validation
                #################################################################################
                print("=" * 10, "VALIDATING", "=" * 10)
                print(models['rgb'].module.temperature)
                with torch.no_grad():
                    for k, valloader in valloaders.items():
                        for i_val, (images_list, labels_list, aux_list) in tqdm(enumerate(valloader)):

                            inputs, labels = parseEightCameras(images_list, labels_list, aux_list, device)

                            # Read batch from only one camera
                            bs = cfg['training']['batch_size']
                            images_val = {m: inputs[m][:bs, :, :, :] for m in cfg["models"].keys()}
                            labels_val = labels[:bs, :, :]

                            if labels_val.shape[0] <= 1:
                                continue

                            # Run Models
                            mean = {}
                            variance = {}
                            val_loss = {}

                            for m in cfg["models"].keys():
                                if hasattr(models[m].module, 'forwardMCDO'):
                                    mean[m], variance[m] = models[m].module.forwardMCDO(images_val[m], cfg["recal"])
                                else:
                                    mean[m] = models[m](images_val[m])
                                    variance[m] = torch.zeros(mean[m].shape)
                                val_loss[m] = loss_fn(input=mean[m], target=labels_val)

                            # Fusion Type
                            if cfg["fusion"] == "None":
                                outputs = mean[list(cfg["models"].keys())[0]]
                            elif cfg["fusion"] == "SoftmaxMultiply":
                                outputs = mean["rgb"] * mean["d"]
                            elif cfg["fusion"] == "SoftmaxAverage":
                                outputs = torch.nn.functional.normalize(mean["rgb"]) + torch.nn.functional.normalize(
                                    mean["d"])
                            elif cfg["fusion"] == "WeightedVariance":
                                rgb_var = 1 / (variance["rgb"] + 1e-5)
                                d_var = 1 / (variance["d"] + 1e-5)
                                plt.figure()
                                plt.title("rgb output variance (w/ blackoutNoise)")
                                plt.hist(variance["rgb"].reshape(-1).data.cpu(), bins=50)
                                plt.savefig("rgb.png")

                                plt.figure()
                                plt.hist(variance["d"].reshape(-1).data.cpu(), bins=50)
                                plt.savefig("d.png")
                                exit()

                                print(variance["d"])
                                outputs = (mean["rgb"] * rgb_var) / (rgb_var + d_var) + \
                                          (mean["d"] * d_var) / (rgb_var + d_var)
                            else:
                                print("Fusion Type Not Supported")

                            # plot ground truth vs mean/variance of outputs
                            pred = outputs.argmax(1).cpu().numpy()
                            gt = labels_val.data.cpu().numpy()

                            if i_val % cfg["training"]["png_frames"] == 0:
                                plotPrediction(logdir, cfg, n_classes, i, i_val, k, inputs, pred, gt)
                                for m in cfg["models"].keys():
                                    plotMeansVariances(logdir, cfg, n_classes, i, i_val, m, k + "/" + m, inputs,
                                                       pred, gt, mean[m], variance[m])

                            running_metrics_val[k].update(gt, pred)

                            for m in cfg["models"].keys():
                                val_loss_meter[m][k].update(val_loss[m].item())

                    for m in cfg["models"].keys():
                        for k in valloaders.keys():
                            writer.add_scalar('loss/val_loss/{}/{}'.format(m, k), val_loss_meter[m][k].avg, i + 1)
                            logger.info("%s %s Iter %d Loss: %.4f" % (m, k, i + 1, val_loss_meter[m][k].avg))

                for env, valloader in valloaders.items():
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

                # save best model
                for m in optimizers.keys():
                    model = models[m]
                    optimizer = optimizers[m]
                    scheduler = schedulers[m]

                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(writer.file_writer.get_logdir(),
                                                 "{}_{}_{}_best_model.pkl".format(
                                                     m,
                                                     cfg['model']['arch'],
                                                     cfg['data']['dataset']))
                        torch.save(state, save_path)
            
            if i >= cfg["training"]["train_iters"]:
                break

def parseEightCameras(images, labels, aux, device):
    # Stack 8 Cameras into 1 for MCDO Dataset Testing
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    aux = torch.cat(aux, 0)

    images = images.to(device)
    labels = labels.to(device)

    if len(aux.shape) < len(images.shape):
        aux = aux.unsqueeze(1).to(device)
        depth = torch.cat((aux, aux, aux), 1)
    else:
        aux = aux.to(device)
        depth = torch.cat((aux[:, 0, :, :].unsqueeze(1),
                           aux[:, 1, :, :].unsqueeze(1),
                           aux[:, 2, :, :].unsqueeze(1)), 1)

    fused = torch.cat((images, aux), 1)

    rgb = torch.cat((images[:, 0, :, :].unsqueeze(1),
                     images[:, 1, :, :].unsqueeze(1),
                     images[:, 2, :, :].unsqueeze(1)), 1)

    inputs = {"rgb": rgb,
              "d": depth,
              "rgbd": fused,
              "fused": fused}

    return inputs, labels


def plotPrediction(logdir, cfg, n_classes, i, i_val, k, inputs, pred, gt):
    fig, axes = plt.subplots(3, 4)
    [axi.set_axis_off() for axi in axes.ravel()]

    gt_norm = gt[0, :, :].copy()
    pred_norm = pred[0, :, :].copy()

    # Ensure each mask has same min and max value for matplotlib normalization
    gt_norm[0, 0] = 0
    gt_norm[0, 1] = n_classes
    pred_norm[0, 0] = 0
    pred_norm[0, 1] = n_classes

    axes[0, 0].imshow(inputs['rgb'][0, :, :, :].permute(1, 2, 0).cpu().numpy()[:, :, 0])
    axes[0, 0].set_title("RGB")

    axes[0, 1].imshow(inputs['d'][0, :, :, :].permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title("D")

    axes[0, 2].imshow(gt_norm)
    axes[0, 2].set_title("GT")

    axes[0, 3].imshow(pred_norm)
    axes[0, 3].set_title("Pred")

    # axes[2,0].imshow(conf[0,:,:])
    # axes[2,0].set_title("Conf")

    # if len(cfg['models'])>1:
    #     if cfg['models']['rgb']['learned_uncertainty'] == 'yes':            
    #         channels = int(mean_outputs['rgb'].shape[1]/2)

    #         axes[1,1].imshow(mean_outputs['rgb'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
    #         axes[1,1].set_title("Aleatoric (RGB)")

    #         axes[1,2].imshow(mean_outputs['d'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
    #         # axes[1,2].imshow(mean_outputs['rgb'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
    #         axes[1,2].set_title("Aleatoric (D)")

    #     else:
    #         channels = int(mean_outputs['rgb'].shape[1])

    #     if cfg['models']['rgb']['mcdo_passes']>1:
    #         axes[2,1].imshow(var_outputs['rgb'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
    #         axes[2,1].set_title("Epistemic (RGB)")

    #         axes[2,2].imshow(var_outputs['d'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
    #         # axes[2,2].imshow(var_outputs['rgb'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
    #         axes[2,2].set_title("Epistemic (D)")

    path = "{}/{}".format(logdir, k)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.tight_layout()
    plt.savefig("{}/{}_{}.png".format(path, i_val, i))
    plt.close(fig)


def plotMeansVariances(logdir, cfg, n_classes, i, i_val, m, k, inputs, pred, gt, mean, variance):

    fig, axes = plt.subplots(4, n_classes // 2 + 1)
    [axi.set_axis_off() for axi in axes.ravel()]

    for c in range(n_classes):
        mean_c = mean[0, c, :, :].cpu().numpy()
        variance_c = variance[0, c, :, :].cpu().numpy()

        # Normarlize Image
        mean_c[0, 0] = 0.0
        mean_c[0, 0] = 1.0
        variance_c[0, 0] = 0.0
        variance_c[0, 0] = 1.0

        axes[2 * (c % 2), c // 2].imshow(mean_c)
        axes[2 * (c % 2), c // 2].set_title(str(c) + " Mean")

        axes[2 * (c % 2) + 1, c // 2].imshow(variance_c)
        axes[2 * (c % 2) + 1, c // 2].set_title(str(c) + " Var")

    axes[-1, -1].imshow(variance[0, :, :, :].mean(0).cpu().numpy())
    axes[-1, -1].set_title("Average Variance")

    path = "{}/{}/{}/{}".format(logdir, "meanvar", m, k)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.tight_layout()
    plt.savefig("{}/{}_{}.png".format(path, i_val, i))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/train/rgbd_BayesianSegnet_0.5_T000.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    # cfg is a  with two-level dictionary ['training','data','model']['batch_size']
    with open(args.config) as fp:
        cfg = defaultdict(lambda: None, yaml.load(fp))

    run_id = cfg["id"]
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    # baseline train (concatenation, warping baselines)
    train(cfg, writer, logger, logdir)
    print('done')
    time.sleep(10)
    writer.close()
