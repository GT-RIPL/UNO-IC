import torch
import numpy as np
import torch.nn as nn

from math import ceil
from torch.autograd import Variable

from ptsemseg import caffe_pb2
from ptsemseg.models.utils import *
from ptsemseg.loss import *

###
# Concrete Dropout Import
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
###

pspnet_specs = {
    "pascal": {
        "n_classes": 21,
        "input_size": (473, 473),
        "block_config": [3, 4, 23, 3],
    },
    "cityscapes": {
        "n_classes": 19,
        "input_size": (713, 713),
        "block_config": [3, 4, 23, 3],
    },
    "ade20k": {
        "n_classes": 150,
        "input_size": (473, 473),
        "block_config": [3, 4, 6, 3],
    },
    "airsim": {
        "n_classes": 9,
        "input_size": (512, 512),
        "block_config": [3, 4, 23, 3],
    },     
}

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        
        out = layer(self._concrete_dropout(x, p))
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
        
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x

class pspnet(nn.Module):

    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105

    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow

    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928

    """

    def __init__(
        self,
        n_classes=21,
        block_config=[3, 4, 23, 3],
        input_size=(473, 473),
        version=None,
        mcdo_passes=1,
        dropoutP=0.1,
        learned_uncertainty="none",
        in_channels=3,
        start_layer="convbnrelu1_1",
        end_layer="classification",
        reduction=1.0,
    ):

        super(pspnet, self).__init__()

        self.block_config = (
            pspnet_specs[version]["block_config"]
            if version is not None
            else block_config
        )
        self.n_classes = (
            pspnet_specs[version]["n_classes"] if version is not None else n_classes
        )
        self.input_size = (
            pspnet_specs[version]["input_size"] if version is not None else input_size
        )

        self.mcdo_passes = mcdo_passes
        self.learned_uncertainty = learned_uncertainty

        self.default_layers = [[  "convbnrelu1_1",                   4,   int(64*reduction), 3, 1, 2, False],
                               [  "convbnrelu1_2",   int(64*reduction),   int(64*reduction), 3, 1, 1, False],
                               [  "convbnrelu1_3",   int(64*reduction),  int(128*reduction), 3, 1, 1, False],
                               [     "res_block2",  int(128*reduction),  int(256*reduction), int(64*reduction), self.block_config[0], 1, 1],
                               [     "res_block3",  int(256*reduction),  int(512*reduction), int(128*reduction), self.block_config[1], 2, 1],
                               [     "res_block4",  int(512*reduction), int(1024*reduction), int(256*reduction), self.block_config[2], 1, 2],
                               ["convbnrelu4_aux", int(1024*reduction),  int(256*reduction), 3, 1, 1, False],
                               [        "aux_cls",  int(256*reduction),                None, self.n_classes, 1, 1, 0],
                               [     "res_block5", int(1024*reduction), int(2048*reduction), int(512*reduction), self.block_config[3], 1, 4],
                               ["pyramid_pooling", int(2048*reduction),                None, [6,3,2,1]],
                               [      "cbr_final", int(4096*reduction),  int(512*reduction), 3, 1, 1, False],
                               [ "classification",  int(512*reduction),                None, self.n_classes, 1, 1, 0]]

        # print(start_layer,end_layer)

        # specify in_channels programmatically
        if in_channels == 0:
            if start_layer == "res_block5":
                match = [row[0] for row in (self.default_layers)].index("convbnrelu4_aux")
                in_channels = int(self.default_layers[match-1][2]*4) # two stacked mean and variance = x4
            elif start_layer == "cbr_final":
                match = [row[0] for row in (self.default_layers)].index("cbr_final")
                in_channels = int(self.default_layers[match][1]*0.5) # two stacked mean and variance = x4
            else:
                match = [row[0] for row in (self.default_layers)].index(start_layer)
                in_channels = int(self.default_layers[match-1][2]*4) # two stacked mean and variance = x4
        if in_channels == -1:
            match = [row[0] for row in (self.default_layers)].index(start_layer)
            in_channels = int(self.default_layers[match-1][2]*1) # forwarded output from previous layer

        # double size of input after learned uncertainty layer (mu,sigma)
        if learned_uncertainty=="double_input":
            # match = [row[0] for row in (self.default_layers)].index(start_layer)
            # in_channels = self.default_layers[match][1]*2
            in_channels *= 2


        # Extract Sub Layers for Fusion
        start_i = 0
        end_i = len(self.default_layers)
        for i,row in enumerate(self.default_layers):
            if start_layer == row[0]:
                start_i = i
            if end_layer == row[0]:
                end_i = i

        self.sub_layers = []
        for i,row in enumerate(self.default_layers):
            if i < start_i:
                self.sub_layers.append([row[0],None])
            elif i == start_i:
                self.sub_layers.append([row[0]]+[in_channels]+row[2:])
            elif i > start_i and i <= end_i:
                self.sub_layers.append(row)
            else:
                self.sub_layers.append([row[0],None])


        # print(in_channels,self.sub_layers)

        for i,row in enumerate(self.default_layers):
            if row[0]=="cbr_final":
                if not row[1] is None and not self.sub_layers[i-1][1] is None:
                    self.sub_layers[i] = [row[0]]+[2*self.sub_layers[i-1][1]]+row[2:]


        self.layer_dict = {row[0]:row[1:] for row in self.sub_layers}

        # print(self.layer_dict)

        self.layers = {}
        self.layers['dropoutMCDO'] = nn.Dropout2d(p=dropoutP, inplace=False)
        self.layers['dropout'] = nn.Dropout2d(p=0.1, inplace=False)
        for k,v in self.layer_dict.items():
            if ("convbnrelu" in k or "cbr_final" in k) and not v[0] is None:
                self.layers[k] = conv2DBatchNormRelu(in_channels=v[0],
                                                     k_size=v[2],
                                                     n_filters=v[1],
                                                     padding=v[3],
                                                     stride=v[4],
                                                     bias=v[5])
                self.layers[k+"_concrete"] = ConcreteDropout()

            if "res_block" in k and not v[0] is None:
                self.layers[k] = residualBlockPSP(v[3],v[0],v[2],v[1],v[4],v[5])
                self.layers[k+"_concrete"] = ConcreteDropout()

            if "pyramid_pooling" in k and not v[0] is None:
                self.layers[k] = pyramidPooling(v[0],v[2])
                self.layers[k+"_concrete"] = ConcreteDropout()

            if ("classification" in k or "aux_cls" in k) and not v[0] is None:
                self.layers[k] = nn.Conv2d(v[0],v[2],v[3],v[4],v[5])
                # self.layers[k+"_concrete"] = ConcreteDropout()


        # set attributes from dictionary
        for k,v in self.layers.items():
            setattr(self,k,v)

        # # Encoder
        # if not self.sub_layers["convbnrelu1_1"][0] is None: 
        #     self.layers["convbnrelu1_1"] = conv2DBatchNormRelu(
        #         in_channels=self.sub_layers[0][1], 
        #         k_size=3, 
        #         n_filters=self.sub_layers[0][1], 
        #         padding=1, 
        #         stride=2, 
        #         bias=False,
        #     )
        # else:
        #     self.convbnrelu1_1 = None

        # self.convbnrelu1_2 = conv2DBatchNormRelu(
        #     in_channels=int(64*reduction), k_size=3, n_filters=int(64*reduction), padding=1, stride=1, bias=False
        # )
        # self.convbnrelu1_3 = conv2DBatchNormRelu(
        #     in_channels=int(64*reduction), k_size=3, n_filters=int(128*reduction), padding=1, stride=1, bias=False
        # )

        # # Vanilla Residual Blocks
        # self.res_block2 = residualBlockPSP(self.block_config[0], int(128*reduction), int(64*reduction), int(256*reduction), 1, 1)
        # self.res_block3 = residualBlockPSP(self.block_config[1], int(256*reduction), int(128*reduction), int(512*reduction), 2, 1)

        # # Dilated Residual Blocks
        # self.res_block4 = residualBlockPSP(self.block_config[2], int(512*reduction), int(256*reduction), int(1024*reduction), 1, 2)
        # self.res_block5 = residualBlockPSP(self.block_config[3], int(1024*reduction), int(512*reduction), int(2048*reduction), 1, 4)

        # # Pyramid Pooling Module
        # self.pyramid_pooling = pyramidPooling(int(2048*reduction), [6, 3, 2, 1])

        # # Final conv layers
        # self.cbr_final = conv2DBatchNormRelu(int(4096*reduction), int(512*reduction), 3, 1, 1, False)
        # self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        # self.classification = nn.Conv2d(int(512*reduction), self.n_classes, 1, 1, 0)

        # # Auxiliary layers for training
        # self.convbnrelu4_aux = conv2DBatchNormRelu(
        #     in_channels=int(1024*reduction), k_size=3, n_filters=int(256*reduction), padding=1, stride=1, bias=False
        # )
        # self.aux_cls = nn.Conv2d(int(256*reduction), self.n_classes, 1, 1, 0)

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

        
    def heteroscedastic_loss(self, true, mean, log_var):
        precision = torch.exp(-log_var)
        return torch.mean(torch.sum(precision * (true - mean)**2 + log_var, 1), 0)


    def forward(self, x, dropout=False):

        # # Turn on training to get weight dropout
        # if self.mcdo_passes>1:
        #     dropout = self.dropoutMCDO
        # else:
        #     dropout = self.dropout

        # if self.training:
        #     dropout.train(mode=True)            
        #     dropout_scalar = 1
        # else:
        #     if self.mcdo_passes>1:
        #         dropout.train(mode=True)                
        #         dropout_scalar = 1-dropout.p
        #     else:
        #         dropout.eval()
        #         dropout_scalar = 1

        inp_shape = x.shape[2:]

        num_concretes = len([x for x in self.layers.keys() if 'concrete' in x])
        regularization = torch.zeros( num_concretes, device=x.device )

        ri = 0

        # H, W -> H/2, W/2
        if 'convbnrelu1_1' in self.layers.keys():
            xprev = x                       
            if self.mcdo_passes == 1:
                x = getattr(self,'convbnrelu1_1')(x) 
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'convbnrelu1_1_concrete')(x,getattr(self,'convbnrelu1_1')) 
                ri += 1

            # x *= dropout_scalar
        if 'convbnrelu1_2' in self.layers.keys():
            xprev = x            
            if self.mcdo_passes == 1:
                x = getattr(self,'convbnrelu1_2')(x) 
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'convbnrelu1_2_concrete')(x,getattr(self,'convbnrelu1_2')) 
                ri += 1

            # x *= dropout_scalar
        if 'convbnrelu1_3' in self.layers.keys():
            xprev = x
            if self.mcdo_passes == 1:
                x = getattr(self,'convbnrelu1_3')(x) 
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'convbnrelu1_3_concrete')(x,getattr(self,'convbnrelu1_3')) 
                ri += 1
            
            # x *= dropout_scalar
            
            x = F.max_pool2d(x, 3, 2, 1)

        # # H/4, W/4 -> H/8, W/8
        if 'res_block2' in self.layers.keys():
            xprev = x                        
            if self.mcdo_passes == 1:
                x = getattr(self,'res_block2')(x)                
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'res_block2_concrete')(x,getattr(self,'res_block2')) 
                ri += 1            

            # x *= dropout_scalar
        if 'res_block3' in self.layers.keys():
            xprev = x                   
            if self.mcdo_passes == 1:
                x = getattr(self,'res_block3')(x)                
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'res_block3_concrete')(x,getattr(self,'res_block3')) 
                ri += 1            

            # x *= dropout_scalar
        if 'res_block4' in self.layers.keys():
            xprev = x
            if self.mcdo_passes == 1:
                x = getattr(self,'res_block4')(x)
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'res_block4_concrete')(x,getattr(self,'res_block4')) 
                ri += 1            

            # x *= dropout_scalar

        if self.training and 'convbnrelu4_aux' in self.layers.keys():  # Auxiliary layers for training
            xprev = x            
            if self.mcdo_passes == 1:
                x_aux = getattr(self,'convbnrelu4_aux')(x)
                x_aux = self.dropout(x_aux)
            else:
                x_aux, regularization[ri] = getattr(self,'convbnrelu4_aux_concrete')(x,getattr(self,'convbnrelu4_aux')) 
                ri += 1       

            # x_aux *= dropout_scalar
            x_aux = getattr(self,'aux_cls')(x_aux)

        if 'res_block5' in self.layers.keys():
            xprev = x           
            if self.mcdo_passes == 1:
                x = getattr(self,'res_block5')(x)
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'res_block5_concrete')(x,getattr(self,'res_block5')) 
                ri += 1            

            # x *= dropout_scalar

        if 'pyramid_pooling' in self.layers.keys():
            xprev = x                       
            if self.mcdo_passes == 1:
                x = getattr(self,'pyramid_pooling')(x)                   
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'pyramid_pooling_concrete')(x,getattr(self,'pyramid_pooling')) 
                ri += 1            

            # x *= dropout_scalar

        if 'cbr_final' in self.layers.keys():
            xprev = x            
            if self.mcdo_passes == 1:
                x = getattr(self,'cbr_final')(x)                   
                x = self.dropout(x)
            else:
                x, regularization[ri] = getattr(self,'cbr_final_concrete')(x,getattr(self,'cbr_final')) 
                ri += 1            

            # x *= dropout_scalar

        if 'classification' in self.layers.keys():
            xprev = x
            x = getattr(self,'classification')(x)   
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=True)


        if self.learned_uncertainty == "yes":
            last_module = [s[0] for s in self.sub_layers if not s[1] is None][-1]
            sigma = getattr(self,last_module)(xprev)
            if last_module=="convbnrelu1_3":
                sigma = F.max_pool2d(sigma, 3, 2, 1)
            if last_module=="classification":
                sigma = F.interpolate(sigma, size=self.input_size, mode='bilinear', align_corners=True)

            x = torch.cat((x,sigma),1)

        if self.mcdo_passes > 1:
            if self.training and 'convbnrelu4_aux' in self.layers.keys():
                return (x, x_aux), regularization.sum()
            else:  # eval mode
                return x, regularization.sum()
        else:
            if self.training and 'convbnrelu4_aux' in self.layers.keys():
                return (x, x_aux)
            else:  # eval mode
                return x



            


    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        # My eyes and my heart both hurt when writing this method

        # Only care about layer_types that have trainable parameters
        ltypes = ["BNData", "ConvolutionData", "HoleConvolutionData"]

        def _get_layer_params(layer, ltype):

            if ltype == "BNData":
                gamma = np.array(layer.blobs[0].data)
                beta = np.array(layer.blobs[1].data)
                mean = np.array(layer.blobs[2].data)
                var = np.array(layer.blobs[3].data)
                return [mean, var, gamma, beta]

            elif ltype in ["ConvolutionData", "HoleConvolutionData"]:
                is_bias = layer.convolution_param.bias_term
                weights = np.array(layer.blobs[0].data)
                bias = []
                if is_bias:
                    bias = np.array(layer.blobs[1].data)
                return [weights, bias]

            elif ltype == "InnerProduct":
                raise Exception(
                    "Fully connected layers {}, not supported".format(ltype)
                )

            else:
                raise Exception("Unkown layer type {}".format(ltype))

        net = caffe_pb2.NetParameter()
        with open(model_path, "rb") as model_file:
            net.MergeFromString(model_file.read())

        # dict formatted as ->  key:<layer_name> :: value:<layer_type>
        layer_types = {}
        # dict formatted as ->  key:<layer_name> :: value:[<list_of_params>]
        layer_params = {}

        for l in net.layer:
            lname = l.name
            ltype = l.type
            if ltype in ltypes:
                print("Processing layer {}".format(lname))
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)

        # Set affine=False for all batchnorm modules
        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False

            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        # _no_affine_bn(self)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())

            print(
                "CONV {}: Original {} and trans weights {}".format(
                    layer_name, w_shape, weights.shape
                )
            )

            module.weight.data.copy_(torch.from_numpy(weights).view_as(module.weight))

            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                print(
                    "CONV {}: Original {} and trans bias {}".format(
                        layer_name, b_shape, bias.shape
                    )
                )
                module.bias.data.copy_(torch.from_numpy(bias).view_as(module.bias))

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            bn_module = mother_module[1]

            _transfer_conv(conv_layer_name, conv_module)

            mean, var, gamma, beta = layer_params[conv_layer_name + "/bn"]
            print(
                "BN {}: Original {} and trans weights {}".format(
                    conv_layer_name, bn_module.running_mean.size(), mean.shape
                )
            )
            bn_module.running_mean.copy_(
                torch.from_numpy(mean).view_as(bn_module.running_mean)
            )
            bn_module.running_var.copy_(
                torch.from_numpy(var).view_as(bn_module.running_var)
            )
            bn_module.weight.data.copy_(
                torch.from_numpy(gamma).view_as(bn_module.weight)
            )
            bn_module.bias.data.copy_(torch.from_numpy(beta).view_as(bn_module.bias))

        def _transfer_residual(prefix, block):
            block_module, n_layers = block[0], block[1]

            bottleneck = block_module.layers[0]
            bottleneck_conv_bn_dic = {
                prefix + "_1_1x1_reduce": bottleneck.cbr1.cbr_unit,
                prefix + "_1_3x3": bottleneck.cbr2.cbr_unit,
                prefix + "_1_1x1_proj": bottleneck.cb4.cb_unit,
                prefix + "_1_1x1_increase": bottleneck.cb3.cb_unit,
            }

            for k, v in bottleneck_conv_bn_dic.items():
                _transfer_conv_bn(k, v)

            for layer_idx in range(2, n_layers + 1):
                residual_layer = block_module.layers[layer_idx - 1]
                residual_conv_bn_dic = {
                    "_".join(
                        map(str, [prefix, layer_idx, "1x1_reduce"])
                    ): residual_layer.cbr1.cbr_unit,
                    "_".join(
                        map(str, [prefix, layer_idx, "3x3"])
                    ): residual_layer.cbr2.cbr_unit,
                    "_".join(
                        map(str, [prefix, layer_idx, "1x1_increase"])
                    ): residual_layer.cb3.cb_unit,
                }

                for k, v in residual_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)

        convbn_layer_mapping = {
            "conv1_1_3x3_s2": self.convbnrelu1_1.cbr_unit,
            "conv1_2_3x3": self.convbnrelu1_2.cbr_unit,
            "conv1_3_3x3": self.convbnrelu1_3.cbr_unit,
            "conv5_3_pool6_conv": self.pyramid_pooling.paths[0].cbr_unit,
            "conv5_3_pool3_conv": self.pyramid_pooling.paths[1].cbr_unit,
            "conv5_3_pool2_conv": self.pyramid_pooling.paths[2].cbr_unit,
            "conv5_3_pool1_conv": self.pyramid_pooling.paths[3].cbr_unit,
            "conv5_4": self.cbr_final.cbr_unit,
            "conv4_" + str(self.block_config[2] + 1): self.convbnrelu4_aux.cbr_unit,
        }  # Auxiliary layers for training

        residual_layers = {
            "conv2": [self.res_block2, self.block_config[0]],
            "conv3": [self.res_block3, self.block_config[1]],
            "conv4": [self.res_block4, self.block_config[2]],
            "conv5": [self.res_block5, self.block_config[3]],
        }

        # Transfer weights for all non-residual conv+bn layers
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)

        # Transfer weights for final non-bn conv layer
        # _transfer_conv("conv6", self.classification)
        # _transfer_conv("conv6_1", self.aux_cls)

        # Transfer weights for all residual layers
        for k, v in residual_layers.items():
            _transfer_residual(k, v)

    def tile_predict(self, imgs, include_flip_mode=True):
        """
        Predict by takin overlapping tiles from the image.

        Strides are adaptively computed from the imgs shape
        and input size

        :param imgs: torch.Tensor with shape [N, C, H, W] in BGR format
        :param side: int with side length of model input
        :param n_classes: int with number of classes in seg output.
        """

        side_x, side_y = self.input_size
        n_classes = self.n_classes
        n_samples, c, h, w = imgs.shape
        # n = int(max(h,w) / float(side) + 1)
        n_x = int(h / float(side_x) + 1)
        n_y = int(w / float(side_y) + 1)
        stride_x = (h - side_x) / float(n_x)
        stride_y = (w - side_y) / float(n_y)

        x_ends = [
            [int(i * stride_x), int(i * stride_x) + side_x] for i in range(n_x + 1)
        ]
        y_ends = [
            [int(i * stride_y), int(i * stride_y) + side_y] for i in range(n_y + 1)
        ]

        pred = np.zeros([n_samples, n_classes, h, w])
        count = np.zeros([h, w])

        slice_count = 0
        for sx, ex in x_ends:
            for sy, ey in y_ends:
                slice_count += 1

                imgs_slice = imgs[:, :, sx:ex, sy:ey]
                if include_flip_mode:
                    imgs_slice_flip = torch.from_numpy(
                        np.copy(imgs_slice.cpu().numpy()[:, :, :, ::-1])
                    ).float()

                is_model_on_cuda = next(self.parameters()).is_cuda

                inp = Variable(imgs_slice, volatile=True)
                if include_flip_mode:
                    flp = Variable(imgs_slice_flip, volatile=True)

                if is_model_on_cuda:
                    inp = inp.cuda()
                    if include_flip_mode:
                        flp = flp.cuda()

                psub1 = F.softmax(self.forward(inp), dim=1).data.cpu().numpy()
                if include_flip_mode:
                    psub2 = F.softmax(self.forward(flp), dim=1).data.cpu().numpy()
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.0
                else:
                    psub = psub1

                pred[:, :, sx:ex, sy:ey] = psub
                count[sx:ex, sy:ey] += 1.0

        score = (pred / count[None, None, ...]).astype(np.float32)
        return score / np.expand_dims(score.sum(axis=1), axis=1)


# For Testing Purposes only
if __name__ == "__main__":
    cd = 0
    import os
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    import scipy.misc as m
    from ptsemseg.loader.cityscapes_loader import cityscapesLoader as cl

    psp = pspnet(version="cityscapes")

    # Just need to do this one time
    caffemodel_dir_path = "PATH_TO_PSPNET_DIR/evaluation/model"
    psp.load_pretrained_model(
        model_path=os.path.join(caffemodel_dir_path, "pspnet101_cityscapes.caffemodel")
    )
    # psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet50_ADE20K.caffemodel'))
    # psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet101_VOC2012.caffemodel'))

    # psp.load_state_dict(torch.load('psp.pth'))

    psp.float()
    psp.cuda(cd)
    psp.eval()

    dataset_root_dir = "PATH_TO_CITYSCAPES_DIR"
    dst = cl(root=dataset_root_dir)
    img = m.imread(
        os.path.join(
            dataset_root_dir,
            "leftImg8bit/demoVideo/stuttgart_00/stuttgart_00_000000_000010_leftImg8bit.png",
        )
    )
    m.imsave("cropped.png", img)
    orig_size = img.shape[:-1]
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float64)
    img -= np.array([123.68, 116.779, 103.939])[:, None, None]
    img = np.copy(img[::-1, :, :])
    img = torch.from_numpy(img).float()  # convert to torch tensor
    img = img.unsqueeze(0)

    out = psp.tile_predict(img)
    pred = np.argmax(out, axis=1)[0]
    decoded = dst.decode_segmap(pred)
    m.imsave("cityscapes_sttutgart_tiled.png", decoded)
    # m.imsave('cityscapes_sttutgart_tiled.png', pred)

    checkpoints_dir_path = "checkpoints"
    if not os.path.exists(checkpoints_dir_path):
        os.mkdir(checkpoints_dir_path)
    psp = torch.nn.DataParallel(
        psp, device_ids=range(torch.cuda.device_count())
    )  # append `module.`
    state = {"model_state": psp.state_dict()}
    torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_cityscapes.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_50_ade20k.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_pascalvoc.pth"))
    print("Output Shape {} \t Input Shape {}".format(out.shape, img.shape))
