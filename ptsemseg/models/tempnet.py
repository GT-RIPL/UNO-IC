import torch.nn as nn
from torch.autograd import Variable

from .fusion import *
from ptsemseg.models.segnet_mcdo import *
from ptsemseg.utils import mutualinfo_entropy, plotEverything, plotPrediction


class TempNet(nn.Module):
    def __init__(self,
                 n_classes=21,
                 in_channels=3,
                 mcdo_passes=1,
                 full_mcdo=False,
                 freeze_seg=True,
                 freeze_temp=True,
                 scaling_module='None',
                 pretrained_rgb=None,
                 pretrained_d=None
                 ):
        super(TempNet, self).__init__()

        self.segnet = segnet_mcdo(n_classes=n_classes,
                                  mcdo_passes=mcdo_passes,
                                  dropoutP=0,
                                  full_mcdo=full_mcdo,
                                  in_channels=in_channels,
                                  temperatureScaling=False,
                                  freeze_seg=freeze_seg,
                                  freeze_temp=freeze_temp, )

        if pretrained_rgb is not None:
            self.modality = "rgb"
            self.loadModel(self.segnet, pretrained_rgb)

        elif pretrained_d is not None:
            self.modality = "d"
            self.loadModel(self.segnet, pretrained_d)

        else:
            print("no pretrained given")
            exit()

        # freeze segnet networks
        for param in self.segnet.parameters():
            param.requires_grad = False

        # initialize temp net
        self.layers = {
                "temp_down1": segnetDown2(in_channels, 64),
                "temp_down2": segnetDown2(64, 128),
                "temp_up2": segnetUp2(128, 64),
                "temp_up1": segnetUp2(64, 1)}
        
        self.scale_logits = self._get_scale_module(scaling_module)

    def forward(self, inputs):

        # Freeze batchnorm
        self.segnet.eval()

        # computer logits and uncertainty measures
        up1 = self.segnet.module.forwardMCDO_logits(inputs) #(batch,11,512,512,passes)

        tdown1, tindices_1, tunpool_shape1 = self.layers["temp_down1"](inputs)
        tdown2, tindices_2, tunpool_shape2 = self.layers["temp_down2"](tdown1)
        tup2 = self.layers["temp_up2"](tdown2, tindices_2, tunpool_shape2)
        tup1 = self.layers["temp_up1"](tup2, tindices_1, tunpool_shape1) #[batch,1,512,512]
        #temp = tup1.mean((2,3)).unsqueeze(-1).unsqueeze(-1) #(batch,1,1,1)

        x = up1 * tup1.unsqueeze(-1)
        mean = x.mean(-1) #[batch,classes,512,512]
        mean = x.mean(-1) 
        variance = x.std(-1)
        prob = self.softmaxMCDO(x) #[batch,classes,512,512]
        prob = prob.masked_fill(prob < 1e-9, 1e-9)
        entropy,mutual_info = mutualinfo_entropy(prob)#(batch,512,512)
        mean = self.scale_logits(mean, variance, mutual_info, entropy)
        return mean, variance, entropy, mutual_info,temp_map.squeeze(1)#,temp.view(-1),entropy.mean((1,2)),mutual_info.mean((1,2))

    def loadModel(self, model, path):
        model_pkl = path

        print(path)
        if os.path.isfile(model_pkl):
            pretrained_dict = torch.load(model_pkl)['model_state']
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            model.load_state_dict(pretrained_dict)
        else:
            print("model not found")
            exit()

    def _get_scale_module(self, name, n_classes=11, bias_init=None):

        name = str(name)

        return {
            "temperature": TemperatureScaling(n_classes, bias_init),
            "uncertainty": UncertaintyScaling(n_classes, bias_init),
            "LocalUncertaintyScaling": LocalUncertaintyScaling(n_classes, bias_init),
            "GlobalUncertainty": GlobalUncertaintyScaling(n_classes, bias_init),
            "GlobalLocalUncertainty": GlobalLocalUncertaintyScaling(n_classes, bias_init),
            "GlobalEntropyScaling" : GlobalEntropyScaling( n_classes=11,modality=self.modality,isSpatialTemp=True,bias_init)
            "None": None
        }[name]
