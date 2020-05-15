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
from numpy import linalg as LA 

# from ptsemseg.process_img import generate_noise
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loaders
from ptsemseg.utils import Confidence_Diagram, get_logger, parseEightCameras, plotPrediction, plotMeansVariances, plotEntropy, plotMutualInfo, plotSpatial, save_pred, plotEverything,mutualinfo_entropy,save_stats,plotAll
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.degredations import *
from ptsemseg.fusion import *
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
    # val_loss_meter = {m: {env: averageMeter() for env in loaders['val'].keys()} for m in cfg["models"].keys()}

    fusion_scaling = GlobalScaling(cfg['training_stats'])

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
        model_dict = models[model].state_dict()
        model_pkl = attr['resume']
        # for model_key,model_pkl in model_pkl_dict.items():
        if os.path.isfile(model_pkl):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(model_pkl)
            )
            checkpoint = torch.load(model_pkl)
            pretrained_dict = checkpoint['model_state']
            # pretrained_dict = {}

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}
            print("Model {} parameters,Loaded {} parameters".format(len(model_dict),len(pretrained_dict)))
            # import ipdb;ipdb.set_trace()
            model_dict.update(pretrained_dict)
            models[model].load_state_dict(pretrained_dict, strict=False)
            logger.info("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
            print("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
        else:
            logger.info("No checkpoint found at '{}'".format(model_pkl))
            print("No checkpoint found at '{}'".format(model_pkl))

    stats_dir = '/'.join(logdir.split('/')[:-1])
    log_prior = torch.load(os.path.join(stats_dir,'stats','log_prior.pkl'))
    prior = torch.tensor(log_prior).to('cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
    #if cfg['uncertainty']: #"BayesianGMM" or cfg['fusion'] == 'MixedGMM' or cfg['fusion'] == "FixedBayesianGMM"  or cfg['fusion'] == 'LinearGMM':
    mean_stats = {}
    cov_stats = {}
    cov_fixed = {}
    cov_fixed_inv = {}
    cov_fixed_det = {}
    det_cov = {}
    inv_cov = {}
    for m in cfg["models"].keys(): 
        mean_stats[m] = torch.load(os.path.join(stats_dir,'stats',m,'mean.pkl'))
        cov_stats[m] = torch.load(os.path.join(stats_dir,'stats',m,'cov.pkl'))
        cov_fixed[m] = torch.load(os.path.join(stats_dir,'stats',m,'cov_fixed.pkl'))
        
        cov_fixed[m] = np.delete(cov_fixed[m],[13,14],0)
        cov_fixed[m] = np.delete(cov_fixed[m],[13,14],1)

        cov_fixed_inv[m] = torch.tensor(np.linalg.inv(cov_fixed[m]),device=device)
        cov_fixed_det[m] = torch.tensor(np.linalg.det(cov_fixed[m]),device=device) 


        for i in range(16):
            for j in range(2):
                cov_stats[m][i] = np.delete(cov_stats[m][i],[13,14],j)
            mean_stats[m][i] = np.delete(mean_stats[m][i],[13,14],0)


        det_cov[m] = [torch.zeros((14,14)) for _ in range(16)]
        inv_cov[m] = [torch.zeros((14,14)) for _ in range(16)]

        for i in range(16):
            if i != 13 and i != 14:
                mean_stats[m][i] = torch.tensor(mean_stats[m][i],device=device)
                det_cov[m][i] = torch.tensor(np.linalg.det(cov_stats[m][i]),device=device) 
                inv_cov[m][i] = torch.tensor(np.linalg.inv(cov_stats[m][i]),device=device)

    [models[m].eval() for m in models.keys()]
    #################################################################################
    # Validation
    #################################################################################
    print("=" * 10, "VALIDATING", "=" * 10)

    

    with torch.no_grad():
        confidence = Confidence_Diagram()
        for k, valloader in loaders['val'].items():
            if cfg["save_stats"]:
                temp_dict_per_loader = {}
                entropy_dict_per_loader = {}
                MI_dict_per_loader = {}
                for m in cfg["models"].keys():
                    temp_dict_per_loader[m] = []
                    entropy_dict_per_loader[m] = []
                    MI_dict_per_loader[m]=[]
            for i_val, (input_list, labels_list) in tqdm(enumerate(valloader)):
                # ipdb;ipdb.set_trace()
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
                entropy = {}
                mutual_info = {}
                val_loss = {}
                entropy_ave = {}
                MI_ave = {}
                temp_ave = {}
                DR = {}
                # Inference
                for m in cfg["models"].keys():
                    mean[m], entropy[m], mutual_info[m] = models[m](images_val[m])
                    if cfg['scaling']:
                        if cfg["models"][m]["arch"] == "DeepLab" or "Segnet":
                            DR[m] = fusion_scaling(entropy[m],mutual_info[m],modality = m,mode=cfg['scaling_metrics'])
                            mean[m] = mean[m] * DR[m]
                        else:
                            if 'rgb' not in DR:
                                DR['rgb'] = 1
                            if 'd' not in DR:
                                DR['d'] = 1
                            mean[m] = mean["rgbd"]*torch.min(DR['rgb'],DR['d'])
                        
                entropy_ave[m] = entropy[m].mean((1,2))
                
                mean = uncertainty(mean,cfg,det_cov=det_cov,inv_cov=inv_cov,mean_stats=mean_stats,device=device,log_prior=prior)
                mean = imbalance(mean,cfg,log_prior=log_prior)
                outputs = fusion(mean,cfg)
                
                # import ipdb;ipdb.set_trace()
                # aggregate training stats
                if cfg["save_stats"]:
                    for m in cfg["models"].keys():
                        entropy_dict_per_loader[m].extend(entropy_ave[m].cpu().numpy().tolist())
                        # MI_dict_per_loader[m].extend(MI_ave[m].cpu().numpy().tolist())
                        
                # plot ground truth vs mean/variance of outputs
                # if cfg['fusion'] == None or cfg['fusion'] == "SoftmaxMultiply" or cfg['fusion'] == 'BayesianGMM' or cfg['fusion'] == 'FixedBayesianGMM' or cfg['fusion'] == 'LinearGMM' or cfg['fusion'] == 'MixedGMM':          
                #     gt = labels_val
                #     prob, pred = outputs.max(1)
                #     confidence.aggregate_stats(prob,pred,gt)
                #     # import ipdb;ipdb.set_trace()
                    
                #     if i_val % cfg["training"]["png_frames"] == 0:
                #         plotPrediction(logdir, cfg, n_classes, 0, i_val, k, inputs_display, pred, gt)
                #         for m in cfg["models"].keys():
                #             _,pred_m = torch.nn.Softmax(dim=1)(mean[m]).max(1)
                #             plotPrediction(logdir, cfg, n_classes, 0, i_val, k + "/" + m, inputs_display, pred_m, gt)  
                #else:
                    # outputs = outputs/outputs.sum(1).unsqueeze(1)
                prob, pred = outputs.max(1)
                gt = labels_val
                confidence.aggregate_stats(prob,pred,gt)
                e, _ = mutualinfo_entropy(outputs.unsqueeze(-1))
                if i_val % cfg["training"]["png_frames"] == 0:
                    plotPrediction(logdir, cfg, n_classes, 0, i_val, k, inputs_display, pred, gt)
                    labels = ['entropy', 'probability']
                    values = [e, prob]
                    plotEverything(logdir, 0, i_val, k + "/fused", values, labels)

                    for m in cfg["models"].keys():
                        prob,pred_m = torch.nn.Softmax(dim=1)(mean[m]).max(1)
                        labels = ['mutual_info', 'entropy', 'probability']
                        values = [mutual_info[m], entropy[m], prob]
                        plotPrediction(logdir, cfg, n_classes, 0, i_val, k + "/" + m, inputs_display, pred_m, gt)
                        plotEverything(logdir, 0, i_val, k + "/" + m, values, labels)
                    
                running_metrics_val[k].update(gt.data.cpu().numpy(), pred.cpu().numpy())
            if cfg["save_stats"]:
                save_stats(logdir,entropy_dict_per_loader,k,cfg,"_entropy_")
                # save_stats(logdir,MI_dict_per_loader,k,cfg,"_MutualInfo_")

        confidence.compute_ece()
        confidence.save(logdir)
        confidence.print()
    for env, valloader in loaders['val'].items():
        score, class_iou, class_acc,count = running_metrics_val[env].get_scores()
        for k, v in score.items():
            logger.info('{}: {}'.format(k, v))
            writer.add_scalar('val_metrics/{}/{}'.format(env, k), v,  1)

        for k, v in class_iou.items():
            logger.info('cls_iou_{}: {}'.format(k, v))
            writer.add_scalar('val_metrics/{}/cls_iou_{}'.format(env, k), v, 1)

        for k, v in class_acc.items():
            logger.info('cls_acc_{}: {}'.format(k, v))
            writer.add_scalar('val_metrics/{}/cls_acc{}'.format(env, k), v, 1)

       
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
        "--id",
        nargs="?",
        type=str,
        default=None,
        help="Unique identifier for different runs",
    )

    parser.add_argument(
        "--run",
        nargs="?",
        type=str,
        default="",
        help="Directory to rerun",
    )

    parser.add_argument(
        "--beta",
        nargs="?",
        type=float,
        default= None,
        help="parameter for MixedBayesianGMM",
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
        if args.id:
            cfg['id'] = args.id
        logdir = "runs" +'/'+ args.config.split("/")[2]+'/'+cfg['id']

        # append tag
        
    # import ipdb;ipdb.set_trace()
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

    if args.beta != None:
        cfg['beta'] = args.beta/100

    # validate base model
    validate(cfg, writer, logger, logdir)

    print('done')
    time.sleep(10)
    writer.close()
