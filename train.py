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
from ptsemseg.models.recalibrator import *
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

import matplotlib.pyplot as plt

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
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
        augmentations=data_aug)

    r_loader = data_loader(
        data_path,
        is_transform=True,
        split="recal",
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=0.10,
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
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),) for env in cfg['data']['val_subsplit']}

    n_classes = int(t_loader.n_classes)
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    recalloader = data.DataLoader(r_loader,
                                  batch_size=1, #cfg['training']['batch_size'], 
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

    # Setup Meters
    val_loss_meter = {m:{env:averageMeter() for env in valloaders.keys()} for m in cfg["models"].keys()}
    val_CE_loss_meter = {env:averageMeter() for env in valloaders.keys()}
    val_REG_loss_meter = {env:averageMeter() for env in valloaders.keys()}
    time_meter = averageMeter()

    # Load Recalibration Model
    calibration = {m:{'model':Recalibrator(device),'fit':False} for m in cfg["models"].keys()}
    calibrationPerClass = {m:{n:{'model':Recalibrator(device),'fit':False} for n in range(n_classes)} for m in cfg["models"].keys()}



    start_iter = 0
    models = {}
    optimizers = {}
    schedulers = {}

    # Setup Model
    for model,attr in cfg["models"].items():

        models[model] = get_model(cfg["model"], 
                                  n_classes,
                                  input_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
                                  in_channels=attr['in_channels'],
                                  start_layer=attr['start_layer'],
                                  end_layer=attr['end_layer'],
                                  mcdo_passes=attr['mcdo_passes'], 
                                  dropoutP=attr['dropoutP'],
                                  learned_uncertainty=attr['learned_uncertainty'],
                                  reduction=attr['reduction']).to(device)



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
        if str(attr['resume']) is not "None" and not "caffemodel" in attr['resume']:

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

                # 1. filter out unnecessary keys
                pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (k in model_dict)} # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}

                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 

                # 3. load the new state dict
                models[model].load_state_dict(pretrained_dict)

                if attr['resume']=='same_yaml':
                    # models[model].load_state_dict(checkpoint["model_state"])
                    optimizers[model].load_state_dict(checkpoint["optimizer_state"])
                    schedulers[model].load_state_dict(checkpoint["scheduler_state"])
                    start_iter = checkpoint["epoch"]
                else:
                    start_iter = checkpoint["epoch"] #0

                # start_iter = 0
                logger.info("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
            else:
                logger.info("No checkpoint found at '{}'".format(model_pkl))        





    best_iou = -100.0
    i = start_iter
    flag = True

    while i <= cfg["training"]["train_iters"] and flag:

        #################################################################################
        # Training
        #################################################################################
        for (images_list, labels_list, aux_list) in trainloader:

            inputs, labels = parseEightCameras( images_list, labels_list, aux_list, device )

            m = list(cfg["models"].keys())[0]


            # Read batch from only one camera
            bs = cfg['training']['batch_size']
            images = {m:inputs[m][:bs,:,:,:] for m in cfg["models"].keys()}
            # images = torch.cat((inputs["rgb"][:bs,:,:,:],inputs["d"][:bs,:,:,:]),1)
            labels = labels[:bs,:,:]

            if labels.shape[0]<=1:
                continue

            # images = images_list[0]
            # labels = labels_list[0]
            # images = generate_noise(images,cfg["data"]["noisy_type"])
            #labels = generate_noise(labels,cfg["data"]["noisy_type"])

            i += 1
            start_ts = time.time()
            
            [schedulers[m].step() for m in schedulers.keys()]
            [models[m].train() for m in models.keys()]
            [optimizers[m].zero_grad() for m in optimizers.keys()]

            # Run Models
            output_bp = {}; mean = {}; variance = {}; outputs = {}; loss = {}
            for m in cfg["models"].keys():
                # m = list(cfg["models"].keys())[0]
                output_bp[m], mean[m], variance[m] = models[m](images[m])
                outputs[m] = output_bp[m]

                loss[m] = loss_fn(input=outputs[m], target=labels)
                loss[m].backward()
                optimizers[m].step()




            # outputs = models[m](images)
            # loss = loss_fn(input=outputs, target=labels)
            # loss.backward()
            # [optimizers[m].step() for m in optimizers.keys()]


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
                    writer.add_scalar('loss/train_loss/'+m, loss[m].item(), i+1)
                    # writer.add_scalar('loss/train_CE_loss', CE_loss.item(), i+1)
                    # writer.add_scalar('loss/train_REG_loss', REG_loss, i+1)
                time_meter.reset()
            #################################################################################



            ###  Validation 
            if (i + 1) % cfg["training"]["val_interval"] == 0 or \
               (i + 1) == cfg["training"]["train_iters"]:


                [models[m].eval() for m in models.keys()]

                #################################################################################
                # Recalibration
                #################################################################################
                steps = 50

                ranges = list(zip([1.*a/steps for a in range(steps+2)][:-2],
                                  [1.*a/steps for a in range(steps+2)][1:]))
                                  
                val = ['sumval_pred_in_range',
                       'num_obs_in_range',
                       'num_in_range',
                       'sumval_pred_below_range',
                       'num_obs_below_range',
                       'num_below_range',
                       'num_correct']   

                overall_match_var = {m:{r:{v:0 for v in val} for r in ranges} for m in cfg['models'].keys()}
                per_class_match_var = {m:{r:{c:{v:0 for v in val} for c in range(n_classes)} for r in ranges} for m in cfg['models'].keys()}

                with torch.no_grad():
                    for i_recal, (images_list, labels_list, aux_list) in tqdm(enumerate(recalloader)):
                    
                        inputs, labels = parseEightCameras( images_list, labels_list, aux_list, device )

                        # Read batch from only one camera
                        bs = cfg['training']['batch_size']
                        images_recal = {m:inputs[m][:bs,:,:,:] for m in cfg["models"].keys()}
                        labels_recal = labels[:bs,:,:]

                        # Run Models
                        output_bp = {}; mean = {}; variance = {}; outputs_recal = {}; 
                        for m in cfg["models"].keys():
                            # m = list(cfg["models"].keys())[0]
                            output_bp[m], mean[m], variance[m] = models[m](images_recal[m])
                            outputs_recal[m] = output_bp[m]

                            overall_match_var,per_class_match_var = accumulateEmpirical(overall_match_var,per_class_match_var,ranges,n_classes,m,labels_recal,mean,variance)

                for m in cfg["models"].keys():

                    calibration, calibrationPerClass, overall_match_var, per_class_match_var = fitCalibration(calibration,calibrationPerClass,overall_match_var,per_class_match_var,ranges,n_classes,m,device)


                    for k,v in overall_match_var[m].items():
                        print(k,v["num_in_range"],v["pred"],v["obs"])

                        for c in range(n_classes):
                            print("   ",c,per_class_match_var[m][k][c]["num_in_range"],
                                          per_class_match_var[m][k][c]["pred"],
                                          per_class_match_var[m][k][c]["obs"])

                    showCalibration(calibration,calibrationPerClass,overall_match_var,per_class_match_var,ranges,m,logdir,cfg,n_classes,i,i_recal,device)

                    # predictCalibration(calibration,calibrationPerClass,overall_match_var,per_class_match_var):

                with torch.no_grad():
                    for i_recal, (images_list, labels_list, aux_list) in tqdm(enumerate(recalloader)):
                    
                        inputs, labels = parseEightCameras( images_list, labels_list, aux_list, device )

                        # Read batch from only one camera
                        bs = cfg['training']['batch_size']
                        images_recal = {m:inputs[m][:bs,:,:,:] for m in cfg["models"].keys()}
                        labels_recal = labels[:bs,:,:]

                        # Run Models
                        output_bp = {}; mean = {}; variance = {}; outputs_recal = {}; 
                        for m in cfg["models"].keys():
                            # m = list(cfg["models"].keys())[0]
                            output_bp[m], mean[m], variance[m] = models[m](images_recal[m])
                            outputs_recal[m] = output_bp[m]

                            outputs = outputs_recal[m]

                            pred = outputs.data.max(1)[1].cpu().numpy()
                            gt = labels_recal.data.cpu().numpy()

                            softmax_mu = {}
                            softmax_mu[m] = torch.nn.Softmax(1)(mean[m])
                            plotMeansVariances(logdir,cfg,n_classes,i,i_recal,m,"recal/pre_recal",inputs,pred,gt,mean,softmax_mu)

                            softmax_mu_recal = softmax_mu
                            for c in range(n_classes):
                                softmax_mu_recal[m][:,c,:,:] = calibrationPerClass[m][c]['model'].predict(softmax_mu[m][:,c,:,:].reshape(-1)).reshape(softmax_mu[m][:,c,:,:].shape)

                            plotMeansVariances(logdir,cfg,n_classes,i,i_recal,m,"recal/post_recal",inputs,pred,gt,mean,softmax_mu_recal)

                            pred = softmax_mu_recal[m].data.max(1)[1].cpu().numpy()
                            plotPrediction(logdir,cfg,n_classes,i,i_recal,"recal/"+m,inputs,pred,gt)

                #################################################################################


                #################################################################################
                # Validation
                #################################################################################
                with torch.no_grad():
                    for k,valloader in valloaders.items():
                        for i_val, (images_list, labels_list, aux_list) in tqdm(enumerate(valloader)):

                            inputs, labels = parseEightCameras( images_list, labels_list, aux_list, device )

                            m = list(cfg["models"].keys())[0]

                            # Read batch from only one camera
                            bs = cfg['training']['batch_size']
                            images_val = {m:inputs[m][:bs,:,:,:] for m in cfg["models"].keys()}
                            # images_val = inputs[m][:bs,:,:,:]                                          
                            # images_val = torch.cat((inputs["rgb"][:bs,:,:,:],inputs["d"][:bs,:,:,:]),1)                            
                            labels_val = labels[:bs,:,:]

                            if labels_val.shape[0]<=1:
                                continue

                            # Run Models
                            output_bp = {}; mean = {}; variance = {}; outputs_val = {}; val_loss = {}
                            for m in cfg["models"].keys():
                                # m = list(cfg["models"].keys())[0]
                                output_bp[m], mean[m], variance[m] = models[m](images_val[m])
                                outputs_val[m] = output_bp[m]

                                val_loss[m] = loss_fn(input=outputs_val[m], target=labels_val)

                            # output_bp, mean, variance = models[m](images_val)
                            # outputs = output_bp
                            # val_loss = loss_fn(input=outputs, target=labels_val)

                            # Fusion Type
                            if cfg["fusion"]=="None":
                                outputs = outputs_val[list(cfg["models"].keys())[0]]
                            elif cfg["fusion"]=="scores_variance_weighted":
                                outputs = mean["rgb"]*variance["d"]+mean["d"]*variance["rgb"]                                
                            elif cfg["fusion"]=="scores_even_weighted":
                                outputs = mean["rgb"]*0.5+mean["d"]*0.5
                            elif cfg["fusion"]=="softmax_variance_weighted":
                                outputs = torch.nn.Softmax(1)(mean["rgb"])*variance["d"]+torch.nn.Softmax(1)(mean["d"])*variance["rgb"]
                            elif cfg["fusion"]=="softmax_even_weighted":
                                outputs = torch.nn.Softmax(1)(mean["rgb"])*0.5+torch.nn.Softmax(1)(mean["d"])*0.5
                            else:
                                print("Fusion Type Not Supported")
                                exit()                              

                            pred = outputs.data.max(1)[1].cpu().numpy()
                            gt = labels_val.data.cpu().numpy()


                            pre_recal = {}
                            post_recal = {}
                            for m in cfg["models"].keys():
                                pre_recal[m] = mean[m] #torch.nn.Softmax(1)(mean[m])
                                post_recal[m] = pre_recal[m]

                                # post_recal = pre_recal
                                for c in range(n_classes):
                                    post_recal[m][:,c,:,:] = calibrationPerClass[m][c]['model'].predict(pre_recal[m][:,c,:,:].reshape(-1)).reshape(pre_recal[m][:,c,:,:].shape)


                            fused = post_recal["rgb"]+post_recal["d"]
                            pred = fused.data.max(1)[1].cpu().numpy()                  


                            if i_val % cfg["training"]["png_frames"] == 0:
                                plotPrediction(logdir,cfg,n_classes,i,i_val,k,inputs,pred,gt)

                                for m in cfg["models"]:
                                    # plotMeansVariances(logdir,cfg,n_classes,i,i_val,m,k,inputs,pred,gt,mean,variance)

                                    plotMeansVariances(logdir,cfg,n_classes,i,i_val,m,k+"/pre_recal_meanvar",inputs,pred,gt,mean,pre_recal)
                                    plotMeansVariances(logdir,cfg,n_classes,i,i_val,m,k+"/post_recal_meanvar",inputs,pred,gt,mean,post_recal)

                                    pre_pred = pre_recal[m].data.max(1)[1].cpu().numpy()
                                    plotPrediction(logdir,cfg,n_classes,i,i_val,k+"/"+m+"/pre_recal_pred",inputs,pre_pred,gt)
                                    post_pred = post_recal[m].data.max(1)[1].cpu().numpy()
                                    plotPrediction(logdir,cfg,n_classes,i,i_val,k+"/"+m+"/post_recal_pred",inputs,post_pred,gt)

                            


                            running_metrics_val[k].update(gt, pred)

                            for m in cfg["models"].keys():
                                val_loss_meter[m][k].update(val_loss[m].item())
                                # val_CE_loss_meter[k].update(CE_loss.item())
                                # val_REG_loss_meter[k].update(REG_loss)



                    for m in cfg["models"].keys():
                        for k in valloaders.keys():
                            writer.add_scalar('loss/val_loss/{}/{}'.format(m,k), val_loss_meter[m][k].avg, i+1)
                            # writer.add_scalar('loss/val_CE_loss/{}'.format(k), val_CE_loss_meter[k].avg, i+1)
                            # writer.add_scalar('loss/val_REG_loss/{}'.format(k), val_REG_loss_meter[k].avg, i+1)
                            logger.info("%s %s Iter %d Loss: %.4f" % (m, k, i + 1, val_loss_meter[m][k].avg))
                
                for env,valloader in valloaders.items():
                    score, class_iou = running_metrics_val[env].get_scores()
                    for k, v in score.items():
                        print(k, v)
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/{}'.format(env,k), v, i+1)

                    for k, v in class_iou.items():
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/cls_{}'.format(env,k), v, i+1)

                    for m in cfg["models"].keys():
                        val_loss_meter[m][env].reset()
                    running_metrics_val[env].reset()


                for m in optimizers.keys():
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
                #################################################################################

                exit()

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break

def parseEightCameras(images,labels,aux,device):

    # Stack 8 Cameras into 1 for MCDO Dataset Testing
    images = torch.cat(images,0)
    labels = torch.cat(labels,0)
    aux = torch.cat(aux,0)

    images = images.to(device)
    labels = labels.to(device)

    if len(aux.shape)<len(images.shape):
        aux = aux.unsqueeze(1).to(device)
        depth = torch.cat((aux,aux,aux),1)
    else:
        aux = aux.to(device)
        depth = torch.cat((aux[:,0,:,:].unsqueeze(1),
                           aux[:,1,:,:].unsqueeze(1),
                           aux[:,2,:,:].unsqueeze(1)),1)

    fused = torch.cat((images,aux),1)

    rgb = torch.cat((images[:,0,:,:].unsqueeze(1),
                     images[:,1,:,:].unsqueeze(1),
                     images[:,2,:,:].unsqueeze(1)),1)

    inputs = {"rgb": rgb,
              "d": depth,
              "fused": fused}

    return inputs, labels

def plotPrediction(logdir,cfg,n_classes,i,i_val,k,inputs,pred,gt):

    fig, axes = plt.subplots(3,4)
    [axi.set_axis_off() for axi in axes.ravel()]

    gt_norm = gt[0,:,:].copy()
    pred_norm = pred[0,:,:].copy()

    # Ensure each mask has same min and max value for matplotlib normalization
    gt_norm[0,0] = 0
    gt_norm[0,1] = n_classes
    pred_norm[0,0] = 0
    pred_norm[0,1] = n_classes

    axes[0,0].imshow(inputs['rgb'][0,:,:,:].permute(1,2,0).cpu().numpy()[:,:,0])
    axes[0,0].set_title("RGB")

    axes[0,1].imshow(inputs['d'][0,:,:,:].permute(1,2,0).cpu().numpy())
    axes[0,1].set_title("D")

    axes[0,2].imshow(gt_norm)
    axes[0,2].set_title("GT")

    axes[0,3].imshow(pred_norm)
    axes[0,3].set_title("Pred")

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


        


    path = "{}/{}".format(logdir,k)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}/{}_{}.png".format(path,i_val,i))
    plt.close(fig)    

def plotMeansVariances(logdir,cfg,n_classes,i,i_val,m,k,inputs,pred,gt,mean,variance):

    # n_classes = int(mean.shape[1])

    fig, axes = plt.subplots(4,n_classes//2+1)
    [axi.set_axis_off() for axi in axes.ravel()]

    for c in range(n_classes):
        mean_c = mean[m][0,c,:,:].cpu().numpy()
        variance_c = variance[m][0,c,:,:].cpu().numpy()

        # Normarlize Image
        mean_c[0,0] = 0.0
        mean_c[0,0] = 1.0
        variance_c[0,0] = 0.0
        variance_c[0,0] = 1.0


        axes[2*(c%2),c//2].imshow(mean_c)
        axes[2*(c%2),c//2].set_title(str(c)+" Mean")

        axes[2*(c%2)+1,c//2].imshow(variance_c)
        axes[2*(c%2)+1,c//2].set_title(str(c)+" Var")

    axes[-1,-1].imshow(variance[m][0,:,:,:].mean(0).cpu().numpy())
    axes[-1,-1].set_title("Average Variance")

    path = "{}/{}/{}/{}".format(logdir,"meanvar",m,k)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}/{}_{}.png".format(path,i_val,i))
    plt.close(fig)    




if __name__ == "__main__":
    # 
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/segnet_airsim_normal.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)
        # cfg is a  with two-level dictionary ['training','data','model']['batch_size']

    run_id = cfg["id"]
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    # baseline train (concatenation, warping baselines)
    train(cfg, writer, logger, logdir)