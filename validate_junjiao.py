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
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.degredations import *

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
    running_metrics_ssma = runningScore(n_classes)# {env: runningScore(n_classes) for env in loaders['val'].keys()}
    running_metrics_uno = runningScore(n_classes) #{env: runningScore(n_classes) for env in loaders['val'].keys()}
    running_metrics_val = {env: runningScore(n_classes) for env in loaders['val'].keys()}
    # Setup Meters
    val_loss_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}
    val_CE_loss_meter = {env: averageMeter() for env in loaders['val'].keys()}
    val_REG_loss_meter = {env: averageMeter() for env in loaders['val'].keys()}
    variance_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}
    entropy_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}
    mutual_info_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}
    time_meter = averageMeter()

    # set seeds for training
    random_seed(cfg['seed'], True)

    start_iter = 0
    models = {}
    swag_models = {}
    optimizers = {}
    schedulers = {}

    # Setup Model
    for model, attr in cfg["models"].items():

        attr = defaultdict(lambda: None, attr)

        models[model] = get_model(name=attr['arch'],
                                  modality = model,
                                  n_classes=n_classes,
                                  input_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
                                  in_channels=attr['in_channels'],
                                  mcdo_passes=attr['mcdo_passes'],
                                  dropoutP=attr['dropoutP'],
                                  full_mcdo=attr['full_mcdo'],
                                  device=device,
                                  temperatureScaling=cfg['temperatureScaling'],
                                  freeze_seg=cfg['freeze_seg'],
                                  freeze_temp=cfg['freeze_temp'],
                                  pretrained_rgb=cfg['pretrained_rgb'],
                                  pretrained_d=cfg['pretrained_d'],
                                  fusion_module=cfg['fusion_module'],
                                  scaling_module=cfg['scaling_module']).to(device)


        models[model] = torch.nn.DataParallel(models[model], device_ids=range(torch.cuda.device_count()))

        # Load pretrained weights
        # if str(attr['resume']) != "None" or str(attr['resume_temp']) != "None" :
        model_dict = models[model].state_dict()
        
        model_pkl_dict = {}

        if attr['resume'] != "None":
            model_pkl_dict["single"]= attr['resume']
            #if attr['resume'] == 'same_yaml':
            #    model_pkl = "{}/{}_pspnet_airsim_best_model.pkl".format(logdir, model)

        if attr['resume_temp'] != "None":
            model_pkl_dict["temp"] = attr['resume_temp']

        if attr['resume_temp_d'] != "None":
            model_pkl_dict["temp_d"] = attr['resume_temp_d']

        if attr['resume_temp_rgb'] != "None":
            model_pkl_dict["temp_rgb"] = attr['resume_temp_rgb']

        # import ipdb;ipdb.set_trace()

            #import ipdb;ipdb.set_trace()

        for model_key,model_pkl in model_pkl_dict.items():
            if os.path.isfile(model_pkl):
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(model_pkl)
                )
                checkpoint = torch.load(model_pkl)

                pretrained_dict_temp = torch.load(model_pkl)['model_state']
                pretrained_dict = {}
                # import ipdb;ipdb.set_trace()
                if model_key == "temp":
                    for wieghts_key,weights in pretrained_dict_temp.items():
                        if wieghts_key.split('.')[1]=='segnet':
                            pretrained_dict[wieghts_key] = weights
                        else:
                            pretrained_dict['module.tempnet'+wieghts_key.split('module')[1]] = weights
                elif model_key == "comp":
                    for wieghts_key,weights in pretrained_dict_temp.items():
                        if wieghts_key.split('.')[1]=='segnet':
                            pretrained_dict[wieghts_key] = weights
                        else:
                            pretrained_dict['module.compnet'+wieghts_key.split('module')[1]] = weights 
                elif model_key == "temp_d":
                    for wieghts_key,weights in pretrained_dict_temp.items():
                        if wieghts_key.split('.')[1]!='segnet':
                            #pretrained_dict[wieghts_key] = weights
                        #else:
                            pretrained_dict['module.tempnet_d'+wieghts_key.split('module')[1]] = weights

                elif model_key == "temp_rgb":
                    for wieghts_key,weights in pretrained_dict_temp.items():
                        if wieghts_key.split('.')[1]!='segnet':
                            #pretrained_dict[wieghts_key] = weights
                        #else:
                            pretrained_dict['module.tempnet_rgb'+wieghts_key.split('module')[1]] = weights
                else:
                    pretrained_dict = pretrained_dict_temp
                #import ipdb;ipdb.set_trace()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                        k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}
                print("Model {} parameters,Loaded {} parameters".format(len(model_dict),len(pretrained_dict)))
                #import ipdb;ipdb.set_trace()
                model_dict.update(pretrained_dict)
                models[model].load_state_dict(pretrained_dict, strict=False)
                if attr['resume'] == 'same_yaml':
                    optimizers[model].load_state_dict(checkpoint["optimizer_state"])
                    schedulers[model].load_state_dict(checkpoint["scheduler_state"])
                    start_iter = checkpoint["epoch"]
                else:
                    start_iter = checkpoint["epoch"]
                logger.info("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
                print("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
            else:
                logger.info("No checkpoint found at '{}'".format(model_pkl))
                print("No checkpoint found at '{}'".format(model_pkl))

    best_iou = -100.0
    i = start_iter
    print("start iteration:{}".format(i))

    [models[m].eval() for m in models.keys()]
    #################################################################################
    # Validation
    #################################################################################
    print("=" * 10, "VALIDATING", "=" * 10)

    with torch.no_grad():
        for k, valloader in loaders['val'].items():
            temp_dict_per_loader = {}
            entropy_dict_per_loader = {}
            MI_dict_per_loader = {}
            for m in cfg["models"].keys():
                temp_dict_per_loader[m] = []
                entropy_dict_per_loader[m] = []
                MI_dict_per_loader[m]=[]
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
                mean_comp = {}
                variance = {}
                entropy = {}
                mutual_info = {}
                val_loss = {}
                temp_map = {}
                comp_map = {}
                entropy_ave = {}
                MI_ave = {}
                temp_ave = {}
                DR = {}
                for m in cfg["models"].keys():

                    if cfg['swa']:
                        mean[m] = swag_models[m](images_val[m])
                        variance[m] = torch.zeros(mean[m].shape)
                        entropy[m] = torch.zeros(labels_val.shape)
                        mutual_info[m] = torch.zeros(labels_val.shape)
                    elif hasattr(models[m].module, 'forwardMCDO'):
                        mean[m], variance[m], entropy[m], mutual_info[m],entropy_ave[m],MI_ave[m] = models[m].module.forwardMCDO_junjiao(images_val[m])
                    elif cfg["models"][m]["arch"] == "tempnet":
                        if m == 'rgb':
                            m_temp = 'd'
                        else:
                            m_temp = 'rgb'
                        mean[m], entropy[m], mutual_info[m], temp_map[m],temp_ave[m],entropy_ave[m],MI_ave[m],DR[m] = models[m](images_val[m],images_val[m_temp],scaling_metrics=cfg['scaling_metrics'])
                    elif cfg["models"][m]["arch"] == "SSMA":    
                        #mean[m], entropy[m], mutual_info[m],entropy_ave[m],MI_ave[m],DR[m] = models[m](images_val[m],[DR['rgb'],DR['d']])
                        mean[m], entropy[m], mutual_info[m],entropy_ave[m],MI_ave[m],DR[m] = models[m](images_val[m],0)
                    else:
                        mean[m] = models[m](images_val[m])
                        #variance[m] = torch.zeros(mean[m].shape)
                        entropy[m] = torch.zeros(labels_val.shape)
                        mutual_info[m] = torch.zeros(labels_val.shape)

                # Fusion Type
                #import ipdb;ipdb.set_trace()
                if cfg["fusion"] == "None":
                    # outputs = torch.nn.Softmax(dim=1)(mean[list(cfg["models"].keys())[0]])
                    outputs = torch.nn.Softmax(dim=1)(mean['rgbd'])
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
                elif cfg["fusion"] == "Noisy-Or":
                    #import ipdb;ipdb.set_trace()
                    #comp_map['rgb'] = torch.max(comp_map['rgb'].unsqueeze(1),(DR['d']!=1).float())
                    #comp_map['d'] = torch.max(comp_map['d'].unsqueeze(1),(DR['rgb']!=1).float())
                    #import ipdb;ipdb.set_trace()
                    #outputs = 1 - (1 - torch.nn.Softmax(dim=1)(mean["rgb"]*comp_map['rgb'].unsqueeze(1))) * (1 - torch.nn.Softmax(dim=1)(mean["d"]*comp_map['d'].unsqueeze(1))) #[batch,11,512,512,1]
                    if 'rgbd' in mean:
                        outputs = 1 - (1 - torch.nn.Softmax(dim=1)(mean["rgb"])) * (1 - torch.nn.Softmax(dim=1)(mean["d"])) * (1 - torch.nn.Softmax(dim=1)(mean["rgbd"]*torch.min(DR['rgb'],DR['d']))) #[batch,11,512,512,1]
                    else:
                        outputs = 1 - (1 - torch.nn.Softmax(dim=1)(mean["rgb"])) * (1 - torch.nn.Softmax(dim=1)(mean["d"])) #[batch,11,512,512,1]
                else:
                    print("Fusion Type Not Supported")
                # aggregate training stats
                for m in cfg["models"].keys():
                    entropy_dict_per_loader[m].extend(entropy_ave[m].cpu().numpy().tolist())
                    MI_dict_per_loader[m].extend(MI_ave[m].cpu().numpy().tolist())
                    if cfg["models"][m]["arch"] == "tempnet":
                        temp_dict_per_loader[m].extend(temp_ave[m].cpu().numpy().tolist())
                # plot ground truth vs mean/variance of outputs            
                outputs = outputs/outputs.sum(1).unsqueeze(1)
                prob, pred = outputs.max(1)
                # _,ssma = mean["rgbd"].max(1)

                gt = labels_val
                e, _ = mutualinfo_entropy(outputs.unsqueeze(-1))
                #import ipdb;ipdb.set_trace()
                # if i_val % cfg["training"]["png_frames"] == 0:
                
                    # plotPrediction(logdir, cfg, n_classes, i, i_val, k, inputs_display, pred, gt)

                #import ipdb;ipdb.set_trace()
                # running_metrics_ssma.update(gt.data.cpu().numpy(), ssma.cpu().numpy())
                # running_metrics_uno.update(gt.data.cpu().numpy(), pred.cpu().numpy())
                # score_ssma, _ = running_metrics_ssma.get_scores()
                # score_uno, _ = running_metrics_uno.get_scores()
                # running_metrics_ssma.reset()
                # running_metrics_uno.reset()                
                # stuff = [inputs_display['rgb'], inputs_display['d'], gt,ssma, pred,[miou_ssma,miou_uno]]
                # plotAll(logdir, i, i_val, k, stuff)
                #import ipdb;ipdb.set_trace()
                if i_val % cfg["training"]["png_frames"] == 0:
                    plotPrediction(logdir, cfg, n_classes, i, i_val, k, inputs_display, pred, gt)
                    labels = ['entropy', 'probability']
                    values = [e, prob]
                    plotEverything(logdir, i, i_val, k + "/fused", values, labels)

                    for m in cfg["models"].keys():
                        prob,pred_m = torch.nn.Softmax(dim=1)(mean[m]).max(1)
                        if cfg["models"][m]["arch"] == "tempnet":
                            labels = ['mutual info', 'entropy', 'probability','temperature']
                            values = [mutual_info[m], entropy[m], prob, temp_map[m]]
                        else:
                            labels = ['mutual info', 'entropy', 'probability']
                            values = [mutual_info[m], entropy[m], prob]
                        plotPrediction(logdir, cfg, n_classes, i, i_val, k + "/" + m, inputs_display, pred_m, gt)
                        plotEverything(logdir, i, i_val, k + "/" + m, values, labels)

                running_metrics_val[k].update(gt.data.cpu().numpy(), pred.cpu().numpy())
                #import ipdb;ipdb.set_trace()
               

            # if cfg["models"][m]["arch"] == "tempnet":
            #     save_stats(logdir,temp_dict_per_loader,k,cfg,"_temp_")
            # save_stats(logdir,entropy_dict_per_loader,k,cfg,"_entropy_")
            # save_stats(logdir,MI_dict_per_loader,k,cfg,"_MutualInfo_")

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

    print('done')
    time.sleep(10)
    writer.close()
