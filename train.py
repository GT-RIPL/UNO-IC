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
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def module_hook(mean_outputs,module, grad_input, grad_out):


    print(mean_outputs.keys())

    if isinstance(mean_outputs['rgb'],tuple):
        rgb,_ = mean_outputs['rgb']
        d,_ = mean_outputs['d']
    else:
        rgb = mean_outputs['rgb']
        d = mean_outputs['d']

    print(mean_outputs['rgb'].shape)


    
    print(len(list(grad_input)))
    print(len(list(grad_out)))
    # print(ptr)
    # ptr[0] = grad_input

def tensor_hook(output,grad):

    sigma = torch.cat((output[:,:int(output.shape[1]/2),:,:],output[:,:int(output.shape[1]/2),:,:]),1)

    modified_grad = 0.5*torch.mul(torch.exp(-sigma),grad.pow(2))+0.5*sigma

    return modified_grad


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

    # add training samples to validation sweep
    valloaders = {**valloaders,'train':data.DataLoader(t_loader,
                                                       batch_size=cfg['training']['batch_size'], 
                                                       num_workers=cfg['training']['n_workers'])}

    # Setup Metrics
    running_metrics_val = {env:runningScore(n_classes) for env in v_loader.keys()}
    val_loss_meter = {env:averageMeter() for env in v_loader.keys()}
    

    start_iter = 0
    models = {}
    optimizers = {}
    schedulers = {}

    layers = [  "convbnrelu1_1",
                "convbnrelu1_2",
                "convbnrelu1_3",
                   "res_block2",
                   "res_block3",
                   "res_block4",
                   "res_block5",
              "pyramid_pooling",
                    "cbr_final",
               "classification"]

    for model,attr in cfg["models"].items():
        if "_static" in model:
            start_layer = "convbnrelu1_1"
            end_layer = layers[layers.index(cfg['start_layers'][1])-1]
        elif "fuse" == model:
            # start_layer = cfg['start_layers'][2]
            # end_layer = "classification"
            start_layer = cfg['start_layers'][1]
            end_layer = "classification"
        else:
            # start_layer = cfg['start_layers'][1]
            # end_layer = layers[layers.index(cfg['start_layers'][2])-1]
            start_layer = cfg['start_layers'][0]
            end_layer = layers[layers.index(cfg['start_layers'][1])-1]

        models[model] = get_model(cfg['model'], 
                                  n_classes, 
                                  input_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
                                  in_channels=attr['in_channels'],
                                  start_layer=start_layer,
                                  end_layer=end_layer,
                                  # start_layer=attr['start_layer'],
                                  # end_layer=attr['end_layer'],
                                  mcdo_passes=attr['mcdo_passes'], 
                                  learned_uncertainty=attr['learned_uncertainty'],
                                  reduction=attr['reduction']).to(device)

        # # Load Pretrained PSPNet
        # if cfg['model'] == 'pspnet':
        #     caffemodel_dir_path = "./models"
        #     model.load_pretrained_model(
        #         model_path=os.path.join(caffemodel_dir_path, "pspnet101_cityscapes.caffemodel")
        #     )  


        models[model] = torch.nn.DataParallel(models[model], device_ids=range(torch.cuda.device_count()))

        # Setup optimizer, lr_scheduler and loss function
        optimizer_cls = get_optimizer(cfg)
        optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                            if k != 'name'}

        optimizers[model] = optimizer_cls(models[model].parameters(), **optimizer_params)
        logger.info("Using optimizer {}".format(optimizers[model]))

        schedulers[model] = get_scheduler(optimizers[model], cfg['training']['lr_schedule'])

        loss_fn = get_loss_function(cfg)
        # loss_sig = # Loss Function for Aleatoric Uncertainty
        logger.info("Using loss {}".format(loss_fn))

        # Load pretrained weights
        if attr['resume'] is not None:
            if os.path.isfile(attr['resume']):
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(attr['resume'])
                )
                checkpoint = torch.load(attr['resume'])

                ###
                pretrained_dict = torch.load(attr['resume'])['model_state']
                model_dict = model.state_dict()

                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 
                # 3. load the new state dict
                model.load_state_dict(pretrained_dict)
                ###


                # model.load_state_dict(checkpoint["model_state"])
                # optimizer.load_state_dict(checkpoint["optimizer_state"])
                # scheduler.load_state_dict(checkpoint["scheduler_state"])
                # start_iter = checkpoint["epoch"]
                start_iter = 0
                logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        attr['resume'], checkpoint["epoch"]
                    )
                )
            else:
                logger.info("No checkpoint found at '{}'".format(attr['resume']))        

        # val_loss_meter[model] = averageMeter()
    # val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels, aux) in trainloader:
            i += 1
            start_ts = time.time()

            [schedulers[m].step() for m in models.keys()]
            [models[m].train() for m in models.keys()]

            images = images.to(device)
            labels = labels.to(device)
            aux = aux.unsqueeze(1).to(device)

            fused = torch.cat((images,aux),1)
            depth = torch.cat((aux,aux,aux),1)
            rgb = torch.cat((images[:,0,:,:].unsqueeze(1),
                             images[:,1,:,:].unsqueeze(1),
                             images[:,2,:,:].unsqueeze(1)),1)

            inputs = {"rgb": rgb,
                      "d": depth,
                      "fused": fused}

            if images.shape[0]<=1:
                continue

            ###

            [optimizers[m].zero_grad() for m in models.keys()]

            # outputs = models['fused'](inputs['fused'])


            # outputs = {m:models[m+"_static"](inputs[m]) for m in ['rgb','d']}
            # inputs = outputs


            # register hooks for modifying gradients     
            # mean_outputs = {}       
            # models['fuse'].register_backward_hook(partial(module_hook,mean_outputs))
            # models['rgb'].register_backward_hook(partial(module_hook,mean_outputs))


            outputs = {}
            outputs_aux = {}

            # Start MCDO Style Training
            for m in ['rgb','d']:

                if i>cfg['models'][m]['mcdo_start_iter']:
                    models[m].mcdo_passes = cfg['models'][m]['mcdo_passes']

                    if not cfg['models'][m]['mcdo_backprop']:
                        with torch.no_grad():
                            for mi in range(models[m].mcdo_passes):
                                x = models[m](inputs[m])
                                x_aux = None
                                if isinstance(x,tuple):
                                    x, x_aux = x
                                if not m in outputs:
                                    outputs[m] = x.unsqueeze(-1)
                                    if not x_aux is None:
                                        outputs_aux[m] = x_aux.unsqueeze(-1)
                                else:
                                    outputs[m] = torch.cat((outputs[m], x.unsqueeze(-1)),-1)
                                    if not x_aux is None:
                                        outputs_aux[m] = torch.cat((outputs_aux[m], x_aux.unsqueeze(-1)),-1)
                    else:
                        for mi in range(models[m].mcdo_passes):
                            x = models[m](inputs[m])
                            x_aux = None
                            if isinstance(x,tuple):
                                x, x_aux = x
                            if not m in outputs:
                                outputs[m] = x.unsqueeze(-1)
                                if not x_aux is None:
                                    outputs_aux[m] = x_aux.unsqueeze(-1)
                            else:
                                outputs[m] = torch.cat((outputs[m], x.unsqueeze(-1)),-1)
                                if not x_aux is None:
                                    outputs_aux[m] = torch.cat((outputs_aux[m], x_aux.unsqueeze(-1)),-1)


                else:
                    models[m].mcdo_passes = 1

                    for mi in range(models[m].mcdo_passes):
                        x = models[m](inputs[m])
                        x_aux = None
                        if isinstance(x,tuple):
                            x, x_aux = x
                        if not m in outputs:
                            outputs[m] = x.unsqueeze(-1)
                            if not x_aux is None:
                                outputs_aux[m] = x_aux.unsqueeze(-1)
                        else:
                            outputs[m] = torch.cat((outputs[m], x.unsqueeze(-1)),-1)
                            if not x_aux is None:
                                outputs_aux[m] = torch.cat((outputs_aux[m], x_aux.unsqueeze(-1)),-1)
                
            mean_outputs = {}
            var_outputs = {}
            for m in outputs.keys():
                mean_outputs[m] = outputs[m].mean(-1)
                if models[m].mcdo_passes>1:
                    var_outputs[m] = outputs[m].pow(2).mean(-1)-mean_outputs[m].pow(2)
                else:
                    var_outputs[m] = mean_outputs[m]  

            # with torch.no_grad():
            #     if cfg['models'][m]['mcdo_passes']>1:
            #         var_outputs = {m:outputs[m].std(-1) for m in outputs.keys()}
            #     else:
            #         var_outputs = {m:outputs[m].mean(-1) for m in outputs.keys()}


            if len(outputs_aux)>0:
                mean_outputs_aux = {m:outputs_aux[m].mean(-1) for m in outputs_aux.keys()}
                with torch.no_grad():
                    if models[m].mcdo_passes>1:
                        var_outputs_aux = {m:outputs_aux[m].std(-1) for m in outputs_aux.keys()}
                    else:
                        var_outputs_aux = {m:outputs_aux[m].mean(-1) for m in outputs_aux.keys()}

            # stack outputs from parallel legs
            intermediate = torch.cat(tuple([mean_outputs[m] for m in outputs.keys()]+[var_outputs[m] for m in outputs.keys()]),1)

            # save mean + std outputs to reference for backward_hook loss update
            # ptrs['rgb'] = [10]


            outputs = models['fuse'](intermediate)

            ###
            hooks = {m:mean_outputs[m].register_hook(partial(tensor_hook,mean_outputs[m])) for m in mean_outputs.keys()}


            # register hooks for modifying gradients
            # ptrs = {m:mean_outputs[m] for m in ['fuse']}
            # [models[m].register_backward_hook(partial(module_hook,ptrs[m])) for m in ptrs.keys()]




            # with torch.no_grad():
                # print(mean_outputs['rgb'].cpu().numpy())
                # print(std_outputs['rgb'].cpu().numpy())
                # print(intermediate.cpu().numpy().shape)
                # print(outputs.cpu().numpy())
     
            loss = loss_fn(input=outputs,target=labels)
            loss.backward()

            [hooks[h].remove() for h in hooks.keys()]            

            # print(ptrs['fuse'][0])
            # # print(list(ptrs['fuse'][0])[0].cpu().numpy().shape)

            # print("outputs",outputs.shape)
            # print("intermediate",intermediate.shape)



            # # average gradients
            # for m in ['rgb','d']:
            #     for p in models[m].parameters():
            #         if not p.grad is None:
            #             p.grad *= 1./cfg['models'][m]['mcdo_passes']

            # exit()

            [optimizers[m].step() for m in models.keys()]





            time_meter.update(time.time() - start_ts)
            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'], 
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i+1)
                time_meter.reset()






            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                
                [models[m].eval() for m in models.keys()]

                with torch.no_grad():
                    # for k,valloader in valloaders.items():
                    for k,valloader in valloaders.items():
                        for i_val, (images, labels, aux) in tqdm(enumerate(valloader)):
                            
                            images = images.to(device)
                            labels = labels.to(device)
                            aux = aux.unsqueeze(1).to(device)

                            fused = torch.cat((images,aux),1)
                            depth = torch.cat((aux,aux,aux),1)
                            rgb = torch.cat((images[:,0,:,:].unsqueeze(1),
                                             images[:,1,:,:].unsqueeze(1),
                                             images[:,2,:,:].unsqueeze(1)),1)

                            inputs = {"rgb": rgb,
                                      "d": depth,
                                      "fused": fused}

                            if images.shape[0]<=1:
                                continue

                            # outputs = {m:models[m+"_static"](inputs[m]) for m in ['rgb','d']}
                            # inputs = outputs

                            outputs = {}
                            outputs_aux = {}

                            for m in ['rgb','d']:
                                for mi in range(cfg['models'][m]['mcdo_passes']):
                                    x = models[m](inputs[m])
                                    if not m in outputs:
                                        outputs[m] = x.unsqueeze(-1)
                                    else:
                                        outputs[m] = torch.cat((outputs[m], x.unsqueeze(-1)),-1)
                              

                            mean_outputs = {m:outputs[m].mean(-1) for m in outputs.keys()}
                            std_outputs = {m:outputs[m].mean(-1) for m in outputs.keys()}

                            # stack outputs from parallel legs
                            intermediate = torch.cat(tuple([mean_outputs[m] for m in outputs.keys()]+[std_outputs[m] for m in outputs.keys()]),1)

                            outputs = models['fuse'](intermediate)

                            # outputs = models['fused'](inputs['fused'])

                            val_loss = loss_fn(input=outputs, target=labels)

                            pred = outputs.data.max(1)[1].cpu().numpy()
                            gt = labels.data.cpu().numpy()

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

                            val_loss_meter[k].update(val_loss.item())


                    for k in valloaders.keys():
                        writer.add_scalar('loss/val_loss', val_loss_meter[k].avg, i+1)
                        logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter[k].avg))
                
                for env,valloader in valloaders.items():
                    score, class_iou = running_metrics_val[env].get_scores()
                    for k, v in score.items():
                        print(k, v)
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/{}'.format(env,k), v, i+1)

                    for k, v in class_iou.items():
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/cls_{}'.format(env,k), v, i+1)

                    val_loss_meter.reset()
                    running_metrics_val[env].reset()

                for m in models.keys():
                    model = models[m]
                    optimizer = optimizers[m]
                    scheduler = schedulers[m]

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
                                                 "{}_{}_{}_best_model.pkl".format(
                                                     m,
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

    run_id = "_".join([cfg['id'],
                       "{}x{}".format(cfg['data']['img_rows'],cfg['data']['img_cols']),
                       "{}reduction".format(cfg['models']['rgb']['reduction']),
                       # "_{}-{}-{}_".format(cfg['start_layers'][0],cfg['start_layers'][1],cfg['start_layers'][2]),
                       "_{}-{}_".format(cfg['start_layers'][0],cfg['start_layers'][1]),
                       # "_rgbS{}->{}_".format(cfg['models']['rgb_static']['start_layer'],cfg['models']['rgb_static']['end_layer']),
                       # "_rgbMCDO{}->{}_".format(cfg['models']['rgb']['start_layer'],cfg['models']['rgb']['end_layer']),
                       # "_fuseS{}->{}_".format(cfg['models']['fuse']['start_layer'],cfg['models']['fuse']['end_layer']),
                       "{}passes".format(cfg['models']['rgb']['mcdo_passes']),
                       "{}mcdostart".format(cfg['models']['rgb']['mcdo_start_iter']),
                       "pretrain" if not cfg['models']['rgb']['resume'] is None else "fromscratch", 
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
