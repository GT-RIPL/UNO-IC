import copy
import logging
from ptsemseg.loss.loss import *


logger = logging.getLogger('ptsemseg')

key2loss = {'CE': CrossEntropy,
            'Focal': FocalLoss,
            'LDAM': LDAMLoss}

def get_loss_function(cfg,weights,cls_num_list):
    if cfg['training']['loss'] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d
    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']
        loss_params = {k:v for k,v in loss_dict[loss_name].items()}
        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))
        elif loss_name == 'LDAM':
            loss_params['cls_num_list'] = cls_num_list
        loss_params['weight'] = weights
        

        logger.info('Using {} with {} params'.format(loss_name, 
                                                     loss_params))
        return key2loss[loss_name](**loss_params)