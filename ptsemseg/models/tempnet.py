import torch.nn as nn
from torch.autograd import Variable

from .fusion import *
from ptsemseg.models.segnet_mcdo import *
from ptsemseg.utils import mutualinfo_entropy, plotEverything, plotPrediction


class TempNet(nn.Module):
    def __init__(self,
                 backbone="segnet",
                 n_classes=21,
                 in_channels=3,
                 mcdo_passes=1,
                 dropoutP=0.1,
                 full_mcdo=False,
                 temperatureScaling=False,
                 freeze_seg=True,
                 freeze_temp=True,
                 pretrained_rgb=None,
                 pretrained_d=None
                 ):
        super(TempNet, self).__init__()

        self.segnet = segnet_mcdo(n_classes=n_classes,
                                  mcdo_passes=mcdo_passes,
                                  dropoutP=dropoutP,
                                  full_mcdo=full_mcdo,
                                  in_channels=in_channels,
                                  temperatureScaling=temperatureScaling,
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

    def forward(self, inputs):

        # Freeze batchnorm
        self.segnet.eval()

        # computer logits and uncertainty measures
        mean, variance, entropy, mutual_info = self.segnet.module.forwardMCDO(inputs)

        return mean

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
