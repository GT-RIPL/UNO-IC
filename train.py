import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

def train(cfg, writer, logger):
    
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    # data_loader = get_loader('airsim')
    # data_path = "../../ros/data/airsim"

    mcdo = cfg['uncertainty']['mcdo']

    # t_loader = data_loader(
    #     data_path,
    #     is_transform=True,
    #     split=cfg['data']['train_split'],
    #     img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
    #     augmentations=data_aug)

    # v_loader = data_loader(
    #     data_path,
    #     is_transform=True,
    #     split=cfg['data']['val_split'],
    #     img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        subsplit=cfg['data']['train_subsplit'],
        # img_size=(512,512),
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = {env:data_loader(
        data_path,
        is_transform=True,
        split="val", subsplit=env,
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),) for env in ["fog_000",
                                                                                 # "fog_005",
                                                                                 # "fog_010",
                                                                                 # "fog_020",
                                                                                 "fog_025",
                                                                                 "fog_050",
                                                                                 "fog_100",
                                                                                 "fog_100__depth_noise_mag20",
                                                                                 "fog_100__rgb_noise_mag20"]}


    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    valloaders = {key:data.DataLoader(v_loader[key], 
                                batch_size=cfg['training']['batch_size'], 
                                num_workers=cfg['training']['n_workers']) for key in v_loader.keys()}

    # Setup Metrics
    running_metrics_val = {env:runningScore(n_classes) for env in v_loader.keys()}

    # Setup Model
    # model = get_model(cfg['model'], n_classes, version="airsim").to(device)
    model = get_model(cfg['model'], n_classes, version="airsim_gate").to(device)
    model_rgb = get_model(cfg['model'], n_classes, version="airsim_rgb").to(device)
    model_depth = get_model(cfg['model'], n_classes, version="airsim_depth").to(device)

    # model = get_model(cfg['gate'], n_classes, version="airsim_gate").to(device)
    # model_gate = get_model(cfg['model'], n_classes, version="airsim_gate").to(device)

    # # Load Pretrained PSPNet
    # if cfg['model'] == 'pspnet':
    #     caffemodel_dir_path = "./models"
    #     model.load_pretrained_model(
    #         model_path=os.path.join(caffemodel_dir_path, "pspnet101_cityscapes.caffemodel")
    #     )  


    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model_rgb = torch.nn.DataParallel(model_rgb, device_ids=range(torch.cuda.device_count()))
    model_depth = torch.nn.DataParallel(model_depth, device_ids=range(torch.cuda.device_count()))
    # model_gate = torch.nn.DataParallel(model_gate, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    if cfg['training']['resumeRGB'] is not None:
        if os.path.isfile(cfg['training']['resumeRGB']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resumeRGB'])
            )
            checkpoint = torch.load(cfg['training']['resumeRGB'])
            model_rgb.load_state_dict(checkpoint["model_state"])
            # optimizer.load_state_dict(checkpoint["optimizer_state"])
            # scheduler.load_state_dict(checkpoint["scheduler_state"])
            # start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resumeRGB'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resumeRGB']))

    if cfg['training']['resumeD'] is not None:
        if os.path.isfile(cfg['training']['resumeD']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resumeD'])
            )
            checkpoint = torch.load(cfg['training']['resumeD'])
            model_depth.load_state_dict(checkpoint["model_state"])
            # optimizer.load_state_dict(checkpoint["optimizer_state"])
            # scheduler.load_state_dict(checkpoint["scheduler_state"])
            # start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resumeD'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resumeD']))

    val_loss_meter = averageMeter()
    val_loss_RGB_meter = averageMeter()
    val_loss_D_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels, aux) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()

            model.train()
            model_rgb.train()
            model_depth.train()

            # model_gate.train()
            
            images = images.to(device)
            labels = labels.to(device)
            aux = aux.unsqueeze(1).to(device)

            # fused = torch.cat((images,aux),1)
            depth = torch.cat((aux,aux,aux,aux),1)
            rgb = torch.cat((images[:,0,:,:].unsqueeze(1),
                             images[:,1,:,:].unsqueeze(1),
                             images[:,2,:,:].unsqueeze(1),
                             images[:,0,:,:].unsqueeze(1)),1)

            if images.shape[0]<=1:
                continue

            optimizer.zero_grad()
            # outputs = model(images)
            
            # outputs = model(fused)
            # loss = loss_fn(input=outputs, target=labels)
            # loss.backward()

            # outputs = model(images)
            # outputs_depth = model_depth(aux)


            x1,x1_aux = model_rgb(rgb)
            loss1 = loss_fn(input=(x1,x1_aux), target=labels)
            loss1.backward()
            
            x2,x2_aux = model_depth(depth)
            loss2 = loss_fn(input=(x2,x2_aux), target=labels)
            loss2.backward()

            if mcdo:
                # Multiple Forward Passes
                with torch.no_grad():
                    model_rgb.eval()
                    model_depth.eval()
                    x1n = torch.zeros(list(x1.shape),device=device).unsqueeze(-1)
                    x2n = torch.zeros(list(x2.shape),device=device).unsqueeze(-1)
                    for ii in range(cfg['uncertainty']['passes']):
                        x1 = model_rgb(rgb,mode="dropout")
                        x2 = model_depth(depth,mode="dropout")
                        x1n = torch.cat((x1n,x1.unsqueeze(-1)),-1)
                        x2n = torch.cat((x2n,x2.unsqueeze(-1)),-1)
                    outputs_rgb = x1n.mean(-1)
                    uncertainty_rgb = x1n.std(-1)
                   
                    outputs_depth = x2n.mean(-1)
                    uncertainty_depth = x2n.std(-1)
                    
                    fused = torch.cat((outputs_rgb,uncertainty_rgb,outputs_depth,uncertainty_depth),1)
            else:
                # Single Forward Pass
                with torch.no_grad():
                    model_rgb.eval()
                    model_depth.eval()
                    x1 = model_rgb(rgb)
                    x2 = model_depth(depth)
                    
                    fused = torch.cat((x1,x2),1)
      

            outputs = model(fused)
            loss = loss_fn(input=outputs, target=labels)
            loss.backward()



            optimizer.step()
            
            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss RGB: {:.4f}  Loss D: {:.4f}  Loss Gate: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'], 
                                           loss1.item(),
                                           loss2.item(),
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss_RGB', loss1.item(), i+1)
                writer.add_scalar('loss/train_loss_D', loss2.item(), i+1)
                writer.add_scalar('loss/train_loss_Gate', loss.item(), i+1)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                
                model.eval()                
                model_rgb.eval()
                model_depth.eval()
                # model_gate.eval()

                with torch.no_grad():
                    for k,valloader in valloaders.items():
                        for i_val, (images_val, labels_val, aux_val) in tqdm(enumerate(valloader)):
                            images_val = images_val.to(device)
                            labels_val = labels_val.to(device)
                            aux_val = aux_val.unsqueeze(1).to(device)

                            fused_val = torch.cat((images_val,aux_val),1)
                            depth_val = torch.cat((aux_val,aux_val,aux_val,aux_val),1)
                            rgb_val = torch.cat((images_val[:,0,:,:].unsqueeze(1),
                                                 images_val[:,1,:,:].unsqueeze(1),
                                                 images_val[:,2,:,:].unsqueeze(1),
                                                 images_val[:,0,:,:].unsqueeze(1)),1)

                            if images_val.shape[0]<=1:
                                continue

                            x1 = model_rgb(rgb_val)                           
                            x2 = model_depth(depth_val)

                            val_loss_RGB = loss_fn(input=x1, target=labels_val)
                            val_loss_D = loss_fn(input=x2, target=labels_val)


                            if mcdo:
                                # Multiple Forward Passes
                                with torch.no_grad():
                                    model_rgb.eval()
                                    model_depth.eval()
                                    x1n = torch.zeros(list(x1.shape),device=device).unsqueeze(-1)
                                    x2n = torch.zeros(list(x2.shape),device=device).unsqueeze(-1)
                                    for ii in range(cfg['uncertainty']['passes']):
                                        x1 = model_rgb(rgb_val,mode="dropout")
                                        x2 = model_depth(depth_val,mode="dropout")
                                        x1n = torch.cat((x1n,x1.unsqueeze(-1)),-1)
                                        x2n = torch.cat((x2n,x2.unsqueeze(-1)),-1)
                                    outputs_rgb = x1n.mean(-1)
                                    uncertainty_rgb = x1n.std(-1)
                                   
                                    outputs_depth = x2n.mean(-1)
                                    uncertainty_depth = x2n.std(-1)

                                    fused_val = torch.cat((outputs_rgb,uncertainty_rgb,outputs_depth,uncertainty_depth),1)
                            else:
                                # Single Forward Pass
                                with torch.no_grad():
                                    model_rgb.eval()
                                    model_depth.eval()
                                    x1 = model_rgb(rgb_val)
                                    x2 = model_depth(depth_val)

                                    fused_val = torch.cat((x1,x2),1)

        
                            outputs = model(fused_val)

                            val_loss = loss_fn(input=outputs, target=labels_val)


                            pred = outputs.data.max(1)[1].cpu().numpy()
                            gt = labels_val.data.cpu().numpy()

                            # print(uncertainty_rgb.shape)

                            # plt.figure()
                            # plt.subplot(221)
                            # plt.imshow(outputs_rgb[0,:3,:,:].permute(1,2,0).cpu().numpy())
                            # plt.subplot(222)
                            # plt.imshow(gt[0,:,:])
                            # plt.subplot(223)
                            # plt.imshow(pred[0,:,:])
                            # plt.subplot(224)
                            # plt.imshow(uncertainty_rgb.mean(1)[0,:,:].cpu().numpy())
                            # plt.show()


                            running_metrics_val[k].update(gt, pred)
                            val_loss_RGB_meter.update(val_loss_RGB.item())
                            val_loss_D_meter.update(val_loss_D.item())
                            val_loss_meter.update(val_loss.item())

                writer.add_scalar('loss/val_loss_RGB', val_loss_RGB_meter.avg, i+1)
                writer.add_scalar('loss/val_loss_D', val_loss_D_meter.avg, i+1)
                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                logger.info("Iter %d Loss RGB: %.4f Loss D: %.4f Loss: %.4f" % (i + 1, val_loss_RGB_meter.avg, val_loss_D_meter.avg, val_loss_meter.avg))

                for env,valloader in valloaders.items():

                    score, class_iou = running_metrics_val[env].get_scores()
                    for k, v in score.items():
                        print(k, v)
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/{}'.format(env,k), v, i+1)

                    for k, v in class_iou.items():
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/cls_{}'.format(env,k), v, i+1)

                    val_loss_RGB_meter.reset()
                    val_loss_D_meter.reset()
                    val_loss_meter.reset()
                    running_metrics_val[env].reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(writer.file_writer.get_logdir(),
                                             "{}_{}_best_model.pkl".format(
                                                 cfg['model']['arch'],
                                                 cfg['data']['dataset']))
                    torch.save(state, save_path)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/pspnet_airsim.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = "_".join([#cfg['id'],
                       "mcdo" if cfg['uncertainty']['mcdo'] else "nomcdo",
                       "pretrain" if not cfg['training']['resumeRGB'] is None else "fromscratch", 
                       "{}x{}".format(cfg['data']['img_rows'],cfg['data']['img_cols']),
                       "{}passes".format(cfg['uncertainty']['passes']),
                       "_train_{}_".format(list(cfg['data']['train_subsplit'])[-1]),
                       "_test_all_",
                       "01-16-2019"])

    # run_id = "mcdo_1pass_pretrain_alignedclasses_fused_fog_all_01-16-2019" #random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    train(cfg, writer, logger)
