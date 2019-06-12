import torch.nn as nn
from torch.autograd import Variable

from ptsemseg.models.utils import *
from ptsemseg.models.recalibrator import *
from ptsemseg.models.segnet_mcdo import *


class fused_segnet_mcdo(nn.Module):
    def __init__(self,
                 n_classes=21,
                 in_channels=3,
                 is_unpooling=True,
                 input_size=(473, 473),
                 batch_size=2,
                 version=None,
                 mcdo_passes=1,
                 dropoutP=0.1,
                 full_mcdo=False,
                 start_layer="down1",
                 end_layer="up1",
                 reduction=1.0,
                 device="cpu",
                 recalibrator="None",
                 temperatureScaling="False",
                 bins=0
                 ):
        super(fused_segnet_mcdo, self).__init__()
        print(recalibrator)

        self.rgb_segnet = segnet_mcdo(n_classes, in_channels, is_unpooling, input_size, batch_size, version,
                                      mcdo_passes, dropoutP, full_mcdo, start_layer,
                                      end_layer, reduction, device, recalibrator, temperatureScaling, bins)
        self.d_segnet = segnet_mcdo(n_classes, in_channels, is_unpooling, input_size, batch_size, version,
                                    mcdo_passes, dropoutP, full_mcdo, start_layer,
                                    end_layer, reduction, device, recalibrator, temperatureScaling, bins)

        self.gatedFusion = GatedFusion(n_classes)

    def forward(self, inputs):
        inputs_rgb = inputs[:, :3, :, :]
        inputs_d = inputs[:, 3:, :, :]

        # TODO figure out how to backpropagate the mean of mcdo passes
        mean_rgb = self.rgb_segnet.forwardAvg(inputs_rgb, recalType="None", backprop=True)
        mean_d = self.d_segnet.forwardAvg(inputs_d, recalType="None", backprop=True)

        print(mean_rgb.shape, mean_d.shape)

        x = self.gatedFusion(mean_rgb, mean_d)

        return x
