import logging
from .losses import *
import torch

def get_loss_function(cfg,**kwargs):
    logger = logging.getLogger("ptsemseg")
    if cfg["training"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        return torch.nn.CrossEntropyLoss(**kwargs)
    else:
        loss_dict = cfg["training"]["loss"]
        loss_type = loss_dict["name"]
        if loss_type is not None and loss_type != 'CrossEntropy' and loss_type != 'SoftIOU':
            loss_params = {k: v for k, v in loss_dict[loss_type].items() if k != "name"}
        else:
            loss_params = {}
        if loss_type == 'CrossEntropy':
            criterion = torch.nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            criterion = FocalLoss(**kwargs)
        else:
            raise NotImplementedError
        return criterion
