"""
Misc Utility functions
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import torch
import torch.nn.functional as F
import copy
import math
import tqdm
import os
import logging
import datetime
import numpy as np
import gc
import pandas as pd
from pandas import DataFrame

from pandas import DataFrame
from collections import OrderedDict

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i:i + n].view(tensor.shape))
        i += n
    return outList


def LogSumExp(x, dim=0):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer, cuda=True, regression=False, verbose=False, subset=None):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print('Stage %d/10. Loss: %12.4f. Acc: %6.2f' % (
                verb_stage + 1, loss_sum / num_objects_current,
                correct / num_objects_current * 100.0
            ))
            verb_stage += 1

    return {
        'loss': loss_sum / num_objects_current,
        'accuracy': None if regression else correct / num_objects_current * 100.0
    }


def eval(loader, model, criterion, cuda=True, regression=False, verbose=False):
    loss_sum = 0.0
    correct = 0.0
    num_objects_total = len(loader.dataset)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)

            if not regression:
                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / num_objects_total,
        'accuracy': None if regression else correct / num_objects_total * 100.0,
    }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            batch_size = input.size(0)
            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += batch_size

    return {
        'predictions': np.vstack(predictions),
        'targets': np.concatenate(targets)
    }


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, m='rgbd', verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():

        for (images_list, labels_list, aux_list) in loader:

            inputs, _ = parseEightCameras(images_list, labels_list, aux_list)
            b = inputs[m].data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(inputs[m], **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps=1e-10):
    return torch.log(x / (1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    # will assume that model is already in eval mode
    # model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


def parseEightCameras(images, labels, aux, device='cuda'):
    # Stack 8 Cameras into 1 for MCDO Dataset Testing
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    aux = torch.cat(aux, 0)

    rgb = images.to(device)
    labels = labels.to(device)
    depth = aux.to(device)

    if len(depth.shape) < len(rgb.shape):
        aux = aux.unsqueeze(1).to(device)
        depth = torch.cat((depth, depth, depth), 1)

    fused = torch.cat((rgb, depth), 1)

    inputs = {"rgb": rgb,
              "d": depth,
              "rgbd": fused,
              "fused": fused}

    return inputs, labels


def plotPrediction(logdir, cfg, n_classes, i, i_val, k,inputs, pred, gt):
    fig, axes = plt.subplots(3, 4)
    [axi.set_axis_off() for axi in axes.ravel()]

    gt_norm = gt[0, :, :].detach().cpu().numpy()
    pred_norm = pred[0, :, :].detach().cpu().numpy()

    # Ensure each mask has same min and max value for matplotlib normalization
    gt_norm[0, 0] = 0
    gt_norm[0, 1] = n_classes
    pred_norm[0, 0] = 0
    pred_norm[0, 1] = n_classes

    # normalize
    axes[0, 0].imshow(inputs['rgb'][0, :, :, :].permute(1, 2, 0).cpu().numpy() / 255)
    axes[0, 0].set_title("RGB")

    axes[0, 1].imshow(inputs['d'][0, :, :, :].permute(1, 2, 0).cpu().numpy() / 255)
    axes[0, 1].set_title("D")

    axes[0, 2].imshow(gt_norm)
    axes[0, 2].set_title("GT")

    axes[0, 3].imshow(pred_norm)
    axes[0, 3].set_title("Pred")

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

    path = "{}/{}".format(logdir, k)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.tight_layout()
    plt.savefig("{}/{}_{}.png".format(path, i_val, i))
    plt.close('all')


def plotMeansVariances(logdir, cfg, n_classes, i, i_val, m, k, inputs, pred, gt, mean, variance):
    fig, axes = plt.subplots(4, n_classes // 2 + 1)
    [axi.set_axis_off() for axi in axes.ravel()]

    for c in range(n_classes):
        mean_c = mean[0, c, :, :].cpu().numpy()
        variance_c = variance[0, c, :, :].cpu().numpy()

        axes[2 * (c % 2), c // 2].imshow(mean_c)
        axes[2 * (c % 2), c // 2].set_title(str(c) + " Mean")

        axes[2 * (c % 2) + 1, c // 2].imshow(variance_c)
        axes[2 * (c % 2) + 1, c // 2].set_title(str(c) + " Var")

    axes[-1, -1].imshow(variance[0, :, :, :].mean(0).cpu().numpy())
    axes[-1, -1].set_title("Average Variance" + str(np.average(variance[0, :, :, :].mean(0).cpu().numpy())))

    path = "{}/{}/{}/{}".format(logdir, "meanvar", m, k)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}/{}_{}.png".format(path, i_val, i))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(variance[0, :, :, :].mean(0).cpu().numpy())
    axes[1].imshow(mean[0, :, :, :].max(0)[0].cpu().numpy())
    plt.savefig("{}/{}_{}avg.png".format(path, i_val, i))
    plt.close('all')

def plotEntropy(logdir, i, i_val, k, pred, variance):
    path = "{}/{}".format(logdir, k)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(variance[0, :, :].cpu().numpy())
    axes[1].imshow(pred[0])
    if not os.path.exists(path):
        os.makedirs(path)
    # import ipdb;ipdb.set_trace()
    plt.savefig("{}/{}_{}_entropy.png".format(path, i_val, i))
    plt.close('all')


def plotMutualInfo(logdir, i, i_val, k, pred, variance):
    path = "{}/{}".format(logdir, k)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(variance[0, :, :].cpu().numpy())
    axes[1].imshow(pred[0])
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}/{}_{}_mutual_info.png".format(path, i_val, i))
    plt.close('all')
    
def plotSpatial(logdir, i, i_val, k, pred, variance):
    path = "{}/{}".format(logdir, k)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(variance[0, :, :].cpu().numpy())
    axes[1].imshow(pred[0])
    if not os.path.exists(path):
        os.makedirs(path)
    plt.colorbar()
    plt.savefig("{}/{}_{}_spatial.png".format(path, i_val, i))
    plt.close('all')

def plotMutualInfoEntropy(logdir, i, i_val, k, pred, variance):
    path = "{}/{}".format(logdir, k)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(pred[0].detach().cpu().numpy())
    axes[1].imshow(variance[0, :, :].detach().cpu().numpy())
    if not os.path.exists(path):
        os.makedirs(path)
    plt.colorbar()
    plt.savefig("{}/{}_{}_mutual_info_entropy.png".format(path, i_val, i))
    plt.close('all')

def plotEverything(logdir, i, i_val, k, values, labels):
    path = "{}/{}".format(logdir, k)
    fig, axes = plt.subplots(len(values)//2+(len(values) % 2),2,squeeze=False)
    #import ipdb; ipdb.set_trace()
        
    for i in range(len(values)):
        axes[i // 2,i % 2 ].set_title(labels[i])
        im = axes[i // 2,i % 2].imshow(values[i][0, :, :].detach().cpu().numpy())
        divider = make_axes_locatable(axes[ i // 2,i % 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        
    if not os.path.exists(path):
        os.makedirs(path)
    fig.tight_layout()
    plt.savefig("{}/{}_{}_everything.png".format(path, i_val, i))
    plt.close('all')

def save_pred(logdir, loc, k, i_val, i, pred, mutual_info, entropy):
    # pred [batch,11,512,512,num_passes]
    # loc [row,col]
    pred = pred.cpu().numpy()
    path = "{}/{}/{}".format(logdir, k, 'dist')
    if not os.path.exists(path):
        os.makedirs(path)
    prediction = {}
    for ps in range(pred.shape[-1]):
        prediction['pass' + str(ps)] = pred[0, :, loc[0], loc[1], ps]
    prediction['Entropy'] = [entropy] + [0] * (pred.shape[1] - 1)
    prediction['mutual_info'] = [mutual_info] + [0] * (pred.shape[1] - 1)

    classes = ['class' + str(cl) for cl in range(pred.shape[1])]
    df = DataFrame(prediction, index=classes)
    # print(df)
    df.to_excel('{}/{}_{}_{}_{}.xlsx'.format(path, i_val, i, loc[0], loc[1]), index=True, header=True)


## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' % (mem_type))
        print('-' * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem))
        print('-' * LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem))
        print('-' * LEN)

    LEN = 65
    print('=' * LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' % ('Element type', 'Size', 'Used MEM(MBytes)'))
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('=' * LEN)



def predictive_entropy(pred):
    # pred [batch,11,512,512,num_passes]
    # return [batch,512,512]
    PEtropy = []
    for b in range(pred.shape[0]):
        avg = pred[b, :, :, :, :].mean(-1)  # [11,512,512]
        entropy = avg * torch.log(avg)  # [11,512,512]
        entropy = -entropy.sum(0)  # [512,512]
        PEtropy.append(entropy.unsqueeze(0))
    PEtropy = torch.cat(PEtropy)
    return PEtropy


def mutual_information(pred):
    # pred [batch,11,512,512,num_passes]
    # return [batch,512,512]
    MI = []
    for b in range(pred.shape[0]):
        avg = pred[b, :, :, :, :].mean(-1)  # [11,512,512]
        entropy = avg * torch.log(avg)  # [11,512,512]
        entropy = -entropy.sum(0)  # [512,512]
        expect = pred[b, :, :, :, :] * torch.log(pred[b, :, :, :, :])  # [11,512,512,10]
        expect = expect.sum(0).mean(-1)  # (512,512)
        MI.append((entropy + expect).unsqueeze(0))
    MI = torch.cat(MI)
    return MI


def mutualinfo_entropy(pred):
    # pred [batch,11,512,512]
    # return [batch,512,512]
    MI = []
    PEtropy = []
    for b in range(pred.shape[0]):
        avg = pred[b, :, :, :, :].mean(-1)  # [11,512,512]
        entropy = avg * torch.log(avg)  # [11,512,512]
        entropy = -entropy.sum(0)  # [512,512]
        PEtropy.append(entropy.unsqueeze(0))
        expect = pred[b, :, :, :, :] * torch.log(pred[b, :, :, :, :])  # [11,512,512,10]
        expect = expect.sum(0).mean(-1)  # (512,512)
        MI.append((entropy + expect).unsqueeze(0))
    PEtropy = torch.cat(PEtropy)
    MI = torch.cat(MI)
    # import ipdb;ipdb.set_trace()
    return PEtropy, MI

def save_pred(logdir,loc,k,i_val,i,pred,mutual_info,entropy):
    #pred [batch,11,512,512,num_passes]
    #loc [row,col]
    pred = pred.cpu().numpy()
    path = "{}/{}/{}".format(logdir,k,'dist')
    if not os.path.exists(path):
        os.makedirs(path)
    prediction = {}
    for ps in range(pred.shape[-1]):
        prediction['pass'+str(ps)] = pred[0,:,loc[0],loc[1],ps]
    prediction['Entropy'] = [entropy] + [0]*(pred.shape[1]-1)
    prediction['mutual_info'] = [mutual_info] + [0]*(pred.shape[1]-1)

    classes = ['class'+str(cl) for cl in range(pred.shape[1])]  
    df = DataFrame(prediction,index=classes)  
    #print(df)
    df.to_excel ('{}/{}_{}_{}_{}.xlsx'.format(path, i_val, i,loc[0],loc[1]), index = True, header=True)



def save_stats(logdir,dict,k,cfg,metric='_temp_'):
    stat = {}
    for m in cfg["models"].keys():
        stat[m+'_'+k+metric+'stats'] = dict[m]
    df = DataFrame(stat) 
    df.to_excel ('{}/{}/rgbd{}stats.xlsx'.format(logdir,k,metric), index = True, header=True)

def plotAll(logdir, i, i_val, k, plot,miou):
    path = "{}/{}/all/".format(logdir, k)
    
    plt.axis('off')
    fig, axes = plt.subplots(1, 1)
    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,1), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,3), colspan=2)
    ax3 = plt.subplot2grid((2,6), (1,0), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,2), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,4), colspan=2)
    
    ax1.imshow(plot[0][0, :, :, :].permute(1, 2, 0).cpu().numpy() / 255)
    ax1.set_title("RGB")
    ax1.set_axis_off()
    
    ax2.imshow(plot[1][0, :, :, :].permute(1, 2, 0).cpu().numpy() / 255)
    ax2.set_title("D")
    ax2.set_axis_off()
    
    ax3.imshow(plot[2][0, :, :].cpu().numpy())
    ax3.set_title("Ground Truth")
    ax3.set_axis_off()
    
    ax4.imshow(plot[3][0, :, :].cpu().numpy())
    ax4.set_title("SSMA: {}".format(miou[0]))
    ax4.set_axis_off()
   
    ax5.imshow(plot[4][0, :, :].cpu().numpy())
    ax5.set_title("UNO++: {}".format(miou[1]))
    ax5.set_axis_off()

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}/{}_{}_all.png".format(path, i_val, i),dpi=300)
    plt.close(fig)


class Confusion_Matrix():
    def __init__(self,logdir,file_name = 'best_model'):
        self.logdir = logdir
        self.confusion_matrix = np.zeros((12,10))
        self.file_name = file_name

    def aggregate_stats(self,predicted,targets):
        # predicted [batch,w,h]
        # & targets [batch,num_class, w,h]
        for i in range(targets.shape[1]):
            mask = predicted == targets[:,i,:,:]

        # for i in range(targets.shape[0]):
        #     self.confusion_matrix[predicted[i],targets[i]] += 1

    def compute_acc(self):  
        self.confusion_matrix[-2,:] = (np.diag(self.confusion_matrix[:-2,:])/self.confusion_matrix[:-2,:].sum(0))*100
        self.confusion_matrix[-1,-1] = (np.diag(self.confusion_matrix[:-2,:]).sum()/self.confusion_matrix[:-2,:].sum())*100
        
    def save(self):
        row_label = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        print(self.confusion_matrix)
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        df = pd.DataFrame(self.confusion_matrix) 
        df.to_excel (os.path.join(self.logdir,self.file_name + '.xlsx'), index = False, header=row_label)

       
    def print(self):
        print(self.confusion_matrix)