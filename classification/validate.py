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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models import get_model
from loader import get_loader
from optimizers import get_optimizer
from loss import get_loss_function
from utils import save_checkpoint, AverageMeter, accuracy, get_logger
from rebalancing import prior_recbalancing

def train(cfg, logger, logdir):
    cudnn.benchmark = True

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["name"])

   
    tloader_params = {k: v for k, v in cfg["data"]["train"].items()}
    tloader_params.update({'root':cfg["data"]["root"]})

    vloader_params = {k: v for k, v in cfg["data"]["val"].items()}
    vloader_params.update({'root':cfg["data"]["root"]})
    
    t_loader = data_loader(**tloader_params)
    v_loader = data_loader(**vloader_params)
    
    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, 
        batch_size=cfg["training"]["batch_size"], 
        num_workers=cfg["training"]["n_workers"]
    )

    # Setup Model
    model = get_model(cfg["model"]["arch"], num_classes=n_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # Resume pre-trained model
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["state_dict"])
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))
    

    
        ##================== Evaluation ============================
        s_prior = np.array(t_loader.get_cls_num_list())
        s_prior = torch.tensor(s_prior/s_prior.sum()).cuda()  
        eval_top1 = AverageMeter('Acc@1', ':6.2f')
        eval_top5 = AverageMeter('Acc@5', ':6.2f')
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(valloader):

                input = input.to(device)
                target = target.to(device)
                logit = model(input)
                # import ipdb;ipdb.set_trace()
                prob = prior_recbalancing(logit,cfg['test']['beta'],s_prior)

                acc1, acc5 = accuracy(prob, target, topk=(1, 5))

                eval_top1.update(acc1[0], input.size(0))
                eval_top5.update(acc5[0], input.size(0))
                
                if i % cfg["training"]["print_interval"] == 0:
                    output = ('Test: [{0}/{1}]\t' 
                              'Prec@1 {top1.avg:.3f}\t'
                              'Prec@5 {top5.avg:.3f}'.format(
                        i, len(valloader), top1=eval_top1, top5=eval_top5))  # TODO
                    print(output)

        output = ('validation Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=eval_top1, top5=eval_top5))
        logger.info(output + '\n')
        print(output)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="/configs/place365.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    logdir = os.path.join("runs","val",cfg["data"]["name"],cfg["model"]["arch"],cfg['id'])
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, logger, logdir)
