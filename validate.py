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
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loaders
from ptsemseg.utils import get_logger, parseEightCameras, plotPrediction, plotEverything, mutualinfo_entropy
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.degredations import *
from ptsemseg.core import likelihood_flattening, prior_recbalancing, fusion
from tensorboardX import SummaryWriter
from collections import defaultdict



def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value) 
    random.seed(seed_value)  
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True  
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
        if os.path.isfile(model_pkl):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(model_pkl)
            )
            checkpoint = torch.load(model_pkl)
            pretrained_dict = checkpoint['model_state']
            # Filter out unnecessary keys
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  
            print("Model {} parameters,Loaded {} parameters".format(len(model_dict),len(pretrained_dict)))
            model_dict.update(pretrained_dict)
            models[model].load_state_dict(pretrained_dict, strict=False)
            logger.info("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
            print("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
        else:
            logger.info("No checkpoint found at '{}'".format(model_pkl))
            print("No checkpoint found at '{}'".format(model_pkl))

    # Load training stats
    stats_dir = '/'.join(logdir.split('/')[:-1])
    prior = torch.load(os.path.join(stats_dir,'stats','prior.pkl'))
    prior = torch.tensor(prior).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device).float() # (1, n_class, 1, 1)
    entropy_stats = torch.load(os.path.join(stats_dir,'stats','entropy.pkl'))
    [models[m].eval() for m in models.keys()]
    #################################################################################
    # Validation
    #################################################################################
    print("=" * 10, "VALIDATING", "=" * 10)

    with torch.no_grad():
        for k, valloader in loaders['val'].items():
            for i_val, (input_list, labels_list) in tqdm(enumerate(valloader)):
                inputs_display, _ = parseEightCameras(input_list['rgb_display'], labels_list, input_list['d_display'], device)
                images_val = {m: input_list[m][0] for m in cfg["models"].keys()}
                labels_val = labels_list[0]
                if labels_val.shape[0] <= 1:
                    continue
                mean = {}
                entropy = {}
                val_loss = {}
                # Inference
                for m in cfg["models"].keys():
                    mean[m], entropy[m] = models[m](images_val[m])
                    mean[m] = likelihood_flattening(mean[m], cfg, entropy[m], entropy_stats, modality = m)
                mean = prior_recbalancing(mean,cfg,prior=prior)
                outputs = fusion(mean,cfg)
        
                prob, pred = outputs.max(1)
                gt = labels_val
                outputs = outputs.masked_fill(outputs < 1e-9, 1e-9)
                e, _ = mutualinfo_entropy(outputs.unsqueeze(-1))
                if i_val % cfg["training"]["png_frames"] == 0:
                    plotPrediction(logdir, cfg, n_classes, 0, i_val,  k + "/fused", inputs_display, pred, gt)
                    labels = ['entropy', 'probability']
                    values = [e, prob]
                    plotEverything(logdir, 0, i_val, k + "/fused", values, labels)

                    for m in cfg["models"].keys():
                        prob,pred_m = torch.nn.Softmax(dim=1)(mean[m]).max(1)
                        labels = [ 'entropy', 'probability']
                        values = [ entropy[m], prob]
                        plotPrediction(logdir, cfg, n_classes, 0, i_val, k + "/" + m, inputs_display, pred_m, gt)
                        plotEverything(logdir, 0, i_val, k + "/" + m, values, labels)
                    
                running_metrics_val[k].update(gt.data.cpu().numpy(), pred.cpu().numpy())
          

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
        "--beta",
        nargs="?",
        type=float,
        default= None,
        help="parameter for MixedBayesianGMM",
    )
    

    args = parser.parse_args()
    
    with open(args.config) as fp:
        cfg = defaultdict(lambda: None, yaml.load(fp))
    if args.id:
        cfg['id'] = args.id
    logdir = "runs" +'/'+ args.config.split("/")[2]+'/'+cfg['id']
    writer = SummaryWriter(logdir)
    logger = get_logger(logdir)
    path = shutil.copy(args.config, logdir)

    if args.beta is not  None:
        cfg['beta'] = args.beta

    # validate base model
    validate(cfg, writer, logger, logdir)
    print('done')
    writer.close()
