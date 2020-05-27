import matplotlib

matplotlib.use('Agg')

import os
import yaml
import time
import shutil
import torch
import random
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loaders
from ptsemseg.utils import get_logger, parseEightCameras, plotPrediction, plotMeansVariances, plotEntropy, plotMutualInfo, mutualinfo_entropy, plotEverything
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.degredations import *
from tensorboardX import SummaryWriter

from functools import partial
from collections import defaultdict
import time
from ptsemseg.utils import bn_update, mem_report

global logdir, cfg, n_classes, i, i_val, k


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def train(cfg, writer, logger, logdir):
    # log git commit
    import subprocess
    label = subprocess.check_output(["git", "describe", "--always"]).strip()
    logger.info("Using commit {}".format(label))
    model,attr = list(cfg['models'].items())[0]
    print("setting up {} model".format(model))
    # Setup seeds
    random_seed(cfg['training']['seed'], True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Setup Dataloader
    loaders, n_classes = get_loaders(cfg["data"]["dataset"], cfg)
    # Setup Metrics
    running_metrics_val = {env: runningScore(n_classes) for env in loaders['val'].keys()}
    # Setup Meters
    val_loss_meter = {env: averageMeter() for env in loaders['val'].keys()}
    variance_meter = {env: averageMeter() for env in loaders['val'].keys()} 
    entropy_meter = {env: averageMeter() for env in loaders['val'].keys()} 
    mutual_info_meter = {env: averageMeter() for env in loaders['val'].keys()}
    time_meter = averageMeter()

    # set seeds for training
    start_iter = 0
    best_iou = -100.0
    attr = defaultdict(lambda: None, attr)
    models = get_model(name=attr['arch'],
                              n_classes=n_classes,
                              input_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
                              in_channels=attr['in_channels'],
                              mcdo_passes=attr['mcdo_passes'],
                              dropoutP=attr['dropoutP'],
                              full_mcdo=attr['full_mcdo'],
                              backbone=attr['backbone'],
                              device=device).to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    optimizers = optimizer_cls(models.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizers))
    models = torch.nn.DataParallel(models, device_ids=range(torch.cuda.device_count()))

    schedulers = get_scheduler(optimizers, cfg['training']['lr_schedule'])

    

    # Load pretrained weights
    if str(attr['resume']) != "None":
        model_pkl = attr['resume']

        if os.path.isfile(model_pkl):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(model_pkl)
            )
            checkpoint = torch.load(model_pkl)
            pretrained_dict = checkpoint['model_state']
            model_dict = models[model].state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            models.load_state_dict(pretrained_dict, strict=False)
            # resume iterations only if specified
            if cfg['training']['resume_iteration']:
                start_iter = checkpoint["epoch"]
            print("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
        else:
            print("No checkpoint found at '{}'".format(model_pkl))
            exit()


    # setup weight for unbalanced dataset
    cls_num_list = np.zeros((n_classes,))
    print("=" * 10, "CALCULATING WEIGHTS", "=" * 10)
    # for _, (_, labels_list) in tqdm(enumerate(loaders['train'])):
    if cfg['training']['reweight'] or cfg['training']['loss']['name']=='LDAM':
        for _, valloader in loaders['val'].items():
            for _, (_, labels_list) in tqdm(enumerate(valloader)):
                for i in range(n_classes):
                    cls_num_list[i] = cls_num_list[i] + (labels_list[0] == i).sum()
        per_cls_weights = np.sort(cls_num_list)[n_classes//2]/cls_num_list
        per_cls_weights[per_cls_weights==np.inf] = 0.0
        per_cls_weights = torch.tensor(per_cls_weights,dtype=torch.float).cuda(device)
    else:
        per_cls_weights = None 
        cls_num_list = None
    import ipdb;ipdb.set_trace()
    loss_fn = get_loss_function(cfg,weights=per_cls_weights,cls_num_list=cls_num_list)
    import ipdb;ipdb.set_trace()
    
    # print("Using loss {}".format(loss_fn))
    i = start_iter
    print("Beginning Training at iteration: {}".format(i))
    while i <= cfg["training"]["train_iters"]:
        #################################################################################
        # Training
        #################################################################################
        print("=" * 10, "TRAINING", "=" * 10)
        for (input_list, labels_list) in loaders['train']:

            
            # if cfg['data']['dataset'] == 'synthia' or cfg['data']['dataset'] == 'airsim':
            # Read batch from only one camera
            inputs, labels = parseEightCameras(input_list['rgb'], labels_list, input_list['d'], device)
            bs = cfg['training']['batch_size']
            images = inputs[model][:bs, :, :, :]
            labels = labels[:bs, :, :]
            # else:
            #     images = input_list.cuda() 
            #     labels = labels_list.cuda()

            if labels.shape[0] <= 1:
                continue

            start_ts = time.time()
            torch.cuda.empty_cache()
            schedulers.step() 
            optimizers.zero_grad()
            models.train()

            # Run Models
            outputs,_,_ = models(images) #images: [2,3,512,512]
            loss = loss_fn(outputs, labels) #labels [2,512,512]
            loss.backward()
            optimizers.step()
            time_meter.update(time.time() - start_ts)
            if (i + 1) % cfg['training']['print_interval'] == 0:
                # for m in cfg["models"].keys():
                fmt_str = "Iter [{:d}/{:d}]  Loss {}: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'],
                                           model,
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss/' + model, loss.item(), i + 1)
                time_meter.reset()

            if i % cfg["training"]["val_interval"] == 0 or i >= cfg["training"]["train_iters"]:
                models.eval()
                print("=" * 10, "VALIDATING", "=" * 10)

                with torch.no_grad():
                    for k, valloader in loaders['val'].items():
                        for i_val, (input_list, labels_list) in tqdm(enumerate(valloader)):

                            if cfg['data']['dataset'] == 'synthia' or cfg['data']['dataset'] == 'airsim':
                            # Read batch from only one camera

                                inputs, labels = parseEightCameras(input_list['rgb'], labels_list, input_list['d'], device)
                                inputs_display, _ = parseEightCameras(input_list['rgb_display'], labels_list,
                                                                      input_list['d_display'], device)
                                bs = cfg['training']['batch_size']
                                images_val = inputs[model][:bs, :, :, :] 
                                labels_val = labels[:bs, :, :]
                            else:
                                inputs = input_list.cuda() 
                                images_val = input_list.cuda() 
                                labels_val = labels_list.cuda()
                                inputs_display ={m: input_list.cuda() for m in ['d','rgb']}

                            if labels_val.shape[0] <= 1:
                                continue
                            # for m in cfg["models"].keys():
                            entropy = torch.zeros(labels_val.shape)
                            mutual_info = torch.zeros(labels_val.shape)
                                
                            mean, entropy, mutual_info = models(images_val)

                            val_loss = loss_fn(mean, labels_val) #labels [2,512,512]
                            # plot ground truth vs mean/variance of outputs
                            outputs = torch.nn.Softmax(dim=1)(mean)
                            prob, pred = outputs.max(1)
                            gt = labels_val
                            e, mi = mutualinfo_entropy(outputs.unsqueeze(-1))

                            if i_val % cfg["training"]["png_frames"] == 0:
                                plotPrediction(logdir, cfg, n_classes, i, i_val, k, inputs_display, pred, gt)
                                labels = ['mutual info', 'entropy', 'probability']
                                values = [mi, e, prob]
                                plotEverything(logdir, i, i_val, k + "/stats", values, labels)

                            running_metrics_val[k].update(gt.cpu().numpy(), pred.cpu().numpy())

                            val_loss_meter[k].update(val_loss.item())
                            entropy_meter[k].update(torch.mean(entropy).item())
                            mutual_info_meter[k].update(torch.mean(mutual_info).item())
                    for k in loaders['val'].keys():
                        writer.add_scalar('loss/val_loss/{}/{}'.format(model, k), val_loss_meter[k].avg, i + 1)
                        logger.info("%s %s Iter %d Loss: %.4f" % (model, k, i, val_loss_meter[k].avg))

                mean_iou = 0

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

                    # for m in cfg["models"].keys():
                    val_loss_meter[env].reset()
                    # variance_meter[env].reset()
                    entropy_meter[env].reset()
                    mutual_info_meter[env].reset()
                    running_metrics_val[env].reset()

                    mean_iou += score["Mean IoU : \t"]

                mean_iou /= len(loaders['val'])

                # save models
                if i <= cfg["training"]["train_iters"]:

                    print("best iou so far: {}, current iou: {}".format(best_iou, mean_iou))

                    # for m in optimizers.keys():
                
                    if not os.path.exists(writer.file_writer.get_logdir() + "/best_model"):
                        os.makedirs(writer.file_writer.get_logdir() + "/best_model")

                    # save best model (averaging the best overall accuracies on the validation set)
                    if mean_iou > best_iou:
                        print('SAVING BEST MODEL')
                        best_iou = mean_iou
                        state = {
                            "epoch": i,
                            "model_state": models.state_dict(),
                            "optimizer_state": optimizers.state_dict(),
                            "scheduler_state": schedulers.state_dict(),
                            "mean_iou": mean_iou,
                        }
                        save_path = os.path.join(writer.file_writer.get_logdir(),
                                                 "best_model",
                                                 "{}_{}_{}_best_model.pkl".format(
                                                     model,
                                                     cfg['models'][model]['arch'],
                                                     cfg['data']['dataset']))
                        torch.save(state, save_path)

                    # save models
                    if 'save_iters' not in cfg['training'].keys() or i % cfg['training']['save_iters'] == 0:
                        state = {
                            "epoch": i,
                            "model_state": models.state_dict(),
                            "optimizer_state": optimizers.state_dict(),
                            "scheduler_state": schedulers.state_dict(),
                            "mean_iou": mean_iou,
                        }
                        save_path = os.path.join(writer.file_writer.get_logdir(),
                                                 "{}_{}_{}_{}_model.pkl".format(
                                                     model,
                                                     cfg['models'][model]['arch'],
                                                     cfg['data']['dataset'],
                                                     i))
                        torch.save(state, save_path)
            i += 1
            if i >= cfg["training"]["train_iters"]:
                break


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
        default="",
        help="Unique identifier for different runs",
    )

    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        default=-1,
        help="Directory to rerun",
    )
    plt.clf()
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = defaultdict(lambda: None, yaml.load(fp))
    if args.id:
        cfg['id'] = args.id
    logdir = "runs" +'/'+ args.config.split("/")[2]+'/'+cfg['id']

    # baseline train (concatenation, warping baselines)
    writer = SummaryWriter(logdir)
    path = shutil.copy(args.config, logdir)
    logger = get_logger(logdir)
    
    # generate seed if none present       
    if cfg['training']['seed'] is None:
        seed = int(time.time())
        cfg['training']['seed'] = seed

        # modify file to reflect seed
        with open(path, 'r') as original:
            data = original.read()
        with open(path, 'w') as modified:
            modified.write("seed: {}\n".format(seed) + data)

    train(cfg, writer, logger, logdir)

    print('done')
    time.sleep(10)
    writer.close()
