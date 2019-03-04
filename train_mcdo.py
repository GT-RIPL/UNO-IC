import matplotlib
matplotlib.use('Agg')

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
plt.ioff()

def tensor_hook(data,grad):
    output, cross_loss = data

    # sigma = output[:,int(output.shape[1]/2):,:,:]
    # grad_mu = grad[:,:int(grad.shape[1]/2),:,:]
    # grad_var = grad[:,int(grad.shape[1]/2):,:,:]

    # modified_grad = torch.cat((0.5*torch.mul(torch.exp(-sigma),grad_mu)+0.5*sigma,grad_var),1)

    sigma = torch.cat((output[:,int(output.shape[1]/2):,:,:],output[:,int(output.shape[1]/2):,:,:]),1)
    grad_mu = torch.cat((grad[:,:int(grad.shape[1]/2),:,:],grad[:,:int(grad.shape[1]/2),:,:]),1)

    # loss = (0.5*torch.mul(torch.exp(-sigma),grad_mu)+0.5*sigma).sum(-1).sum(-1).sum(-1)

    # print(loss.shape)


    # loss = loss.repeat(1,grad.shape[1],grad.shape[2],grad.shape[3])

    # print(grad.shape)
    # print(loss.shape)

    # print(output.shape,sigma.shape,grad.shape,grad_mu.shape)

    # modified_grad = 0.5*torch.mul(torch.exp(-sigma),cross_loss)+0.5*torch.mul(torch.exp(-sigma),grad.pow(2))+0.5*sigma
    # modified_grad = 0.5*torch.mul(torch.exp(-sigma),cross_loss)+0.5*sigma
    # modified_grad = torch.sum(0.5*torch.mul(torch.exp(-sigma),cross_loss)+0.5*sigma,dim=1)
    # modified_grad = 0.5*torch.mul(torch.exp(-sigma),grad)+0.5*sigma
    modified_grad = 0.5*torch.mul(torch.exp(-sigma),grad_mu)+0.5*sigma


    # only apply grad to mean, not also to std


    # modified_grad = loss

    return modified_grad


def train(cfg, writer, logger, logdir):
    
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
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=1.0,
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
        augmentations=data_aug)

    tv_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=0.05,
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = {env:data_loader(
        data_path,
        is_transform=True,
        split="val", subsplits=[env], scale_quantity=cfg['data']['val_reduction'],
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),) for env in [
                                                                                 "8camera_fog_000_dense",
                                                                                 "8camera_rain_dense",
                                                                                ]}

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    valloaders = {key:data.DataLoader(v_loader[key], 
                                      batch_size=cfg['training']['batch_size'], 
                                      num_workers=cfg['training']['n_workers']) for key in v_loader.keys()}

    # add training samples to validation sweep
    valloaders = {**valloaders,'train':data.DataLoader(tv_loader,
                                                       batch_size=cfg['training']['batch_size'], 
                                                       num_workers=cfg['training']['n_workers'])}

    # Setup Metrics
    running_metrics_val = {env:runningScore(n_classes) for env in valloaders.keys()}
    val_loss_meter = {env:averageMeter() for env in valloaders.keys()}
    

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
        if len(cfg['models'])==1:
            start_layer = "convbnrelu1_1"
            end_layer = "classification"
        else:    
            if not cfg['start_layers'] is None:
                if "_static" in model:
                    start_layer = "convbnrelu1_1"
                    end_layer = layers[layers.index(cfg['start_layers'][1])-1]
                elif "fuse" == model:
                    start_layer = cfg['start_layers'][-1]
                    end_layer = "classification"
                else:
                    if len(cfg['start_layers']) == 3:
                        start_layer = cfg['start_layers'][1]
                        end_layer = layers[layers.index(cfg['start_layers'][2])-1]
                    else:
                        start_layer = cfg['start_layers'][0]
                        end_layer = layers[layers.index(cfg['start_layers'][1])-1]
            else:
                start_layer = attr['start_layer']
                end_layer = attr['end_layer']

        print(model,start_layer,end_layer)

        models[model] = get_model(cfg['model'], 
                                  n_classes, 
                                  input_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
                                  in_channels=attr['in_channels'],
                                  start_layer=start_layer,
                                  end_layer=end_layer,
                                  mcdo_passes=attr['mcdo_passes'], 
                                  dropoutP=attr['dropoutP'],
                                  learned_uncertainty=attr['learned_uncertainty'],
                                  reduction=attr['reduction']).to(device)

        # # Load Pretrained PSPNet
        # if cfg['model'] == 'pspnet':
        #     caffemodel_dir_path = "./models"
        #     model.load_pretrained_model(
        #         model_path=os.path.join(caffemodel_dir_path, "pspnet101_cityscapes.caffemodel")
        #     )  

        if "caffemodel" in attr['resume']:
            models[model].load_pretrained_model(model_path=attr['resume'])


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
        if attr['resume'] is not None and not "caffemodel" in attr['resume']:

            model_pkl = attr['resume']
            if attr['resume']=='same_yaml':
                model_pkl = "{}/{}_pspnet_airsim_best_model.pkl".format(logdir,model)

            if os.path.isfile(model_pkl):
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(model_pkl)
                )
                checkpoint = torch.load(model_pkl)

                ###
                pretrained_dict = torch.load(model_pkl)['model_state']
                model_dict = models[model].state_dict()

                # print(model,start_layer,end_layer)


                # 1. filter out unnecessary keys
                pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (k in model_dict)} # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}


                # print(pretrained_dict.keys())

                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 

                print(model_dict.keys())

                # 3. load the new state dict
                models[model].load_state_dict(pretrained_dict)
                ###

                # print(pretrained_dict)


                if attr['resume']=='same_yaml':
                    # models[model].load_state_dict(checkpoint["model_state"])
                    optimizers[model].load_state_dict(checkpoint["optimizer_state"])
                    schedulers[model].load_state_dict(checkpoint["scheduler_state"])
                    start_iter = checkpoint["epoch"]
                else:
                    start_iter = 0
                # start_iter = 0
                logger.info("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
            else:
                logger.info("No checkpoint found at '{}'".format(model_pkl))        

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

            # Stack 8 Cameras into 1 for MCDO Dataset Testing
            images = torch.cat(images,0)
            labels = torch.cat(labels,0)
            aux = torch.cat(aux,0)

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

            reg = torch.zeros(1,device=device )

            if images.shape[0]<=1:
                continue


            [optimizers[m].zero_grad() for m in models.keys()]

            if any("input_fusion" in m for m in models.keys()):
                outputs, _ = models['input_fusion'](inputs['fused'])
            elif any("rgb_only" in m for m in models.keys()):
                outputs, _ = models['rgb_only'](inputs['rgb'])
            elif any("d_only" in m for m in models.keys()):
                outputs, _ = models['d_only'](inputs['d'])

            else:

                outputs = {}
                outputs_aux = {}
                if any("_static" in m for m in models.keys()):                
                    outputs = {m:list(models[m+"_static"](inputs[m]))[0] for m in ['rgb','d']}
                    inputs = outputs

                # Start MCDO Style Training
                outputs = {}
                outputs_aux = {}   
                regs = {}             
                for m in ['rgb','d']:

                    if i>cfg['models'][m]['mcdo_start_iter']:
                        models[m].mcdo_passes = cfg['models'][m]['mcdo_passes']

                        regs[m] = torch.zeros( models[m].mcdo_passes, device=device )

                        if not cfg['models'][m]['mcdo_backprop']:
                            x, regs[m][0] = models[m](inputs[m])
                            x_aux = None
                            if isinstance(x,tuple):
                                x, x_aux = x
                            outputs[m] = x.unsqueeze(-1)
                            if not x_aux is None:
                                outputs_aux[m] = x_aux.unsqueeze(-1)


                            with torch.no_grad():
                                for mi in range(models[m].mcdo_passes-1):
                                    x, regs[m][mi+1] = models[m](inputs[m])
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
                                x, regs[m][mi] = models[m](inputs[m])
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

                        regs[m] = torch.zeros( models[m].mcdo_passes, device=device )

                        for mi in range(models[m].mcdo_passes):
                            x, regs[m][mi] = models[m](inputs[m])
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
                    reg = torch.stack([regs[m].sum() for m in regs.keys()]).sum()

                mean_outputs = {}
                var_outputs = {}
                for m in outputs.keys():
                    mean_outputs[m] = outputs[m].mean(-1)
                    if models[m].mcdo_passes>1:
                        var_outputs[m] = outputs[m].pow(2).mean(-1)-mean_outputs[m].pow(2)
                    else:
                        var_outputs[m] = mean_outputs[m]  

                if len(outputs_aux)>0:
                    mean_outputs_aux = {m:outputs_aux[m].mean(-1) for m in outputs_aux.keys()}
                    with torch.no_grad():
                        if models[m].mcdo_passes>1:
                            var_outputs_aux = {m:outputs_aux[m].std(-1) for m in outputs_aux.keys()}
                        else:
                            var_outputs_aux = {m:outputs_aux[m].mean(-1) for m in outputs_aux.keys()}

                # # UNCERTAINTY
                # # convert log variance to normal variance
                # for m in mean_outputs.keys():
                #     var_split = int(mean_outputs[m].shape[1]/2)
                #     mean_outputs[m][:,:var_split,:,:] = torch.exp(mean_outputs[m][:,:var_split,:,:])


                # stack outputs from parallel legs
                intermediate = torch.cat(tuple([mean_outputs[m] for m in outputs.keys()]+[var_outputs[m] for m in outputs.keys()]),1)

                outputs, _ = models['fuse'](intermediate)

                # auxiliaring training loss
                if len(outputs_aux)>0:
                    outputs = (outputs,*[mean_outputs_aux[m] for m in mean_outputs_aux.keys()])


            # with torch.no_grad():
                # print(mean_outputs['rgb'].cpu().numpy())
                # print(std_outputs['rgb'].cpu().numpy())
                # print(intermediate.cpu().numpy().shape)
                # print(outputs.cpu().numpy())
     
            loss = loss_fn(input=outputs,target=labels) + reg

            # register hooks for modifying gradients for learned uncertainty
            if len(cfg['models'])>1 and cfg['models']['rgb']['learned_uncertainty'] == 'yes':            
                hooks = {m:mean_outputs[m].register_hook(partial(tensor_hook,(mean_outputs[m],loss))) for m in mean_outputs.keys()}

            loss.backward()

            # remove hooks for modifying gradients for learned uncertainty
            if len(cfg['models'])>1 and cfg['models']['rgb']['learned_uncertainty'] == 'yes':            
                [hooks[h].remove() for h in hooks.keys()]            

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
                            
                            # Stack 8 Cameras into 1 for MCDO Dataset Testing
                            images = torch.cat(images,0)
                            labels = torch.cat(labels,0)
                            aux = torch.cat(aux,0)

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

                            reg = torch.zeros(1,device=device )

                            orig = inputs.copy()

                            if images.shape[0]<=1:
                                continue

                            if any("input_fusion" in m for m in models.keys()):
                                outputs, _ = models['input_fusion'](inputs['fused'])

                            elif any("rgb_only" in m for m in models.keys()):
                                outputs, _ = models['rgb_only'](inputs['rgb'])

                            elif any("d_only" in m for m in models.keys()):
                                outputs, _ = models['d_only'](inputs['d'])


                            else:


                                outputs = {}
                                outputs_aux = {}
                                if any("_static" in m for m in models.keys()):                
                                    outputs = {m:list(models[m+"_static"](inputs[m]))[0] for m in ['rgb','d']}
                                    inputs = outputs

                                outputs = {}
                                outputs_aux = {}
                                regs = {}
                                for m in ['rgb','d']:
                                    regs[m] = torch.zeros( models[m].mcdo_passes, device=device )
                                    for mi in range(cfg['models'][m]['mcdo_passes']):
                                        x, regs[m][mi] = models[m](inputs[m])
                                        if not m in outputs:
                                            outputs[m] = x.unsqueeze(-1)
                                        else:
                                            outputs[m] = torch.cat((outputs[m], x.unsqueeze(-1)),-1)
                                reg = torch.stack([regs[m].sum() for m in regs.keys()]).sum()


                                mean_outputs = {m:outputs[m].mean(-1) for m in outputs.keys()}
                                std_outputs = {m:outputs[m].mean(-1) for m in outputs.keys()}

                                # stack outputs from parallel legs
                                intermediate = torch.cat(tuple([mean_outputs[m] for m in outputs.keys()]+[std_outputs[m] for m in outputs.keys()]),1)

                                outputs, _ = models['fuse'](intermediate)




                            val_loss = loss_fn(input=outputs, target=labels) + reg

                            pred = outputs.data.max(1)[1].cpu().numpy()
                            conf = outputs.data.max(1)[0].cpu().numpy()
                            gt = labels.data.cpu().numpy()

                            # Visualization
                            if i_val % cfg['training']['png_frames'] == 0:
                                fig, axes = plt.subplots(3,4)
                                [axi.set_axis_off() for axi in axes.ravel()]

                                axes[0,0].imshow(gt[0,:,:])
                                axes[0,0].set_title("GT")

                                axes[0,1].imshow(orig['rgb'][0,:,:,:].permute(1,2,0).cpu().numpy())
                                axes[0,1].set_title("RGB")

                                axes[0,2].imshow(orig['d'][0,:,:,:].permute(1,2,0).cpu().numpy())
                                axes[0,2].set_title("D")


                                axes[1,0].imshow(pred[0,:,:])
                                axes[1,0].set_title("Pred")

                                axes[2,0].imshow(conf[0,:,:])
                                axes[2,0].set_title("Conf")


                                if len(cfg['models'])>1:
                                    if cfg['models']['rgb']['learned_uncertainty'] == 'yes':            
                                        channels = int(mean_outputs['rgb'].shape[1]/2)

                                        axes[1,1].imshow(mean_outputs['rgb'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
                                        axes[1,1].set_title("Aleatoric (RGB)")

                                        axes[1,2].imshow(mean_outputs['d'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
                                        # axes[1,2].imshow(mean_outputs['rgb'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
                                        axes[1,2].set_title("Aleatoric (D)")

                                    else:
                                        channels = int(mean_outputs['rgb'].shape[1])

                                    if cfg['models']['rgb']['mcdo_passes']>1:
                                        axes[2,1].imshow(std_outputs['rgb'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
                                        axes[2,1].set_title("Epistemic (RGB)")

                                        axes[2,2].imshow(std_outputs['d'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
                                        # axes[2,2].imshow(std_outputs['rgb'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
                                        axes[2,2].set_title("Epistemic (D)")


                                    

                                path = "{}/{}".format(logdir,k)
                                if not os.path.exists(path):
                                    os.makedirs(path)
                                plt.savefig("{}/{}_{}.png".format(path,i_val,i))
                                plt.close(fig)

                            running_metrics_val[k].update(gt, pred)

                            val_loss_meter[k].update(val_loss.item())


                    for k in valloaders.keys():
                        writer.add_scalar('loss/val_loss/{}'.format(k), val_loss_meter[k].avg, i+1)
                        logger.info("%s Iter %d Loss: %.4f" % (k, i + 1, val_loss_meter[k].avg))
                
                for env,valloader in valloaders.items():
                    score, class_iou = running_metrics_val[env].get_scores()
                    for k, v in score.items():
                        print(k, v)
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/{}'.format(env,k), v, i+1)

                    for k, v in class_iou.items():
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/cls_{}'.format(env,k), v, i+1)

                    val_loss_meter[env].reset()
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

    mcdo_model_name = "rgb" if len(cfg['models'])>1 else None #next((s for s in list(cfg['models'].keys()) if "mcdo" in s), None)

    name = [cfg['id']]
    name.append("{}x{}".format(cfg['data']['img_rows'],cfg['data']['img_cols']))
    name.append("_{}_".format("-".join(cfg['start_layers'])))
    name.append("{}bs".format(cfg['training']['batch_size']))
    # name.append("_{}_".format("-".join(cfg['models'].keys())))
    if not mcdo_model_name is None:
        name.append("{}reduction".format(cfg['models'][mcdo_model_name]['reduction']))
        name.append("{}passes".format(cfg['models'][mcdo_model_name]['mcdo_passes']))
        name.append("{}dropoutP".format(cfg['models'][mcdo_model_name]['dropoutP']))
        name.append("{}learnedUncertainty".format(cfg['models'][mcdo_model_name]['learned_uncertainty']))
        name.append("{}mcdostart".format(cfg['models'][mcdo_model_name]['mcdo_start_iter']))
        name.append("{}mcdobackprop".format(cfg['models'][mcdo_model_name]['mcdo_backprop']))
        name.append("pretrain" if not cfg['models'][mcdo_model_name]['resume'] is None else "fromscratch")
    name.append("_train_{}_".format(list(cfg['data']['train_subsplit'])[-1]))
    name.append("_test_all_")
    name.append("01-16-2019")

    run_id = "_".join(name)

    # run_id = "mcdo_1pass_pretrain_alignedclasses_fused_fog_all_01-16-2019" #random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    train(cfg, writer, logger, logdir)
