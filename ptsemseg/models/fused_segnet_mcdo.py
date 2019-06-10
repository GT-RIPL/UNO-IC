import torch.nn as nn
from torch.autograd import Variable

from ptsemseg.models.utils import *
from ptsemseg.models.recalibrator import *
from ptsemseg.models.segnet_mcdo import *


class fused_segnet(nn.Module):
    def __init__(self,
                 n_classes=21,
                 in_channels=3,
                 is_unpooling=True,
                 input_size=(473, 473),
                 batch_size=2,
                 version=None,
                 mcdo_passes=1,
                 fixed_mcdo=False,
                 dropoutP=0.1,
                 learned_uncertainty="none",
                 start_layer="down1",
                 end_layer="up1",
                 reduction=1.0,
                 device="cpu",
                 recalibrator="None",
                 bins=0
                 ):
        super(fused_segnet, self).__init__()

        self.rgb_segnet = segnet_mcdo(n_classes, in_channels, is_unpooling, input_size, batch_size, version,
                                      mcdo_passes, fixed_mcdo, dropoutP, learned_uncertainty, start_layer,
                                      end_layer, reduction, device, recalibrator, bins)
        self.d_segnet = segnet_mcdo(n_classes, in_channels, is_unpooling, input_size, batch_size, version,
                                    mcdo_passes, fixed_mcdo, dropoutP, learned_uncertainty, start_layer,
                                    end_layer, reduction, device, recalibrator, bins)

        self.gatedFusion = GatedFusion(n_classes)

    def forward(self, inputs):
        inputs_rgb = inputs[:, :3, :, :]
        inputs_d = inputs[:, 3:, :, :]
        # TODO figure out how to backpropagate the mean of mcdo passes
        rgb_bp, mean_rgb, var_rgb = self.rgb_segnet(inputs_rgb)
        d_bp, mean_d, var_d = self.d_segnet(inputs_d)

        x = self.gatedFusion(rgb_bp, d_bp)

        return x
