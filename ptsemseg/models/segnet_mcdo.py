import torch.nn as nn
from torch.autograd import Variable

from .fusion.fusion import *
from ptsemseg.models.recalibrator import *
from ptsemseg.utils import mutualinfo_entropy


class segnet_mcdo(nn.Module):
    def __init__(self,
                 modality = 'rgb',
                 n_classes=21,
                 in_channels=3,
                 is_unpooling=True,
                 mcdo_passes=1,
                 dropoutP=0.1,
                 full_mcdo=False,
                 freeze_seg=False,
                 freeze_temp=False,
                 scaling_module = 'None',
                 temperatureScaling=False):
        super(segnet_mcdo, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.mcdo_passes = mcdo_passes
        self.n_classes = n_classes
        self.dropoutP = dropoutP
        self.full_mcdo = full_mcdo
        self.freeze_seg = freeze_seg
        self.freeze_temp = freeze_temp
        self.modality = modality
        
        if not self.full_mcdo:
            self.layers = {
                "down1": segnetDown2(self.in_channels, 64),
                "down2": segnetDown2(64, 128),
                "down3": segnetDown3MCDO(128, 256, pMCDO=dropoutP),
                "down4": segnetDown3MCDO(256, 512, pMCDO=dropoutP),
                "down5": segnetDown3MCDO(512, 512, pMCDO=dropoutP),
                "up5": segnetUp3MCDO(512, 512, pMCDO=dropoutP),
                "up4": segnetUp3MCDO(512, 256, pMCDO=dropoutP),
                "up3": segnetUp3MCDO(256, 128, pMCDO=dropoutP),
                "up2": segnetUp2(128, 64),
                "up1": segnetUp2(64, n_classes, relu=True),
            }
        else:
            self.layers = {
                "down1": segnetDown2MCDO(self.in_channels, 64, pMCDO=dropoutP),
                "down2": segnetDown2MCDO(64, 128, pMCDO=dropoutP),
                "down3": segnetDown3MCDO(128, 256, pMCDO=dropoutP),
                "down4": segnetDown3MCDO(256, 512, pMCDO=dropoutP),
                "down5": segnetDown3MCDO(512, 512, pMCDO=dropoutP),
                "up5": segnetUp3MCDO(512, 512, pMCDO=dropoutP),
                "up4": segnetUp3MCDO(512, 256, pMCDO=dropoutP),
                "up3": segnetUp3MCDO(256, 128, pMCDO=dropoutP),
                "up2": segnetUp2MCDO(128, 64, pMCDO=dropoutP),
                "up1": segnetUp2MCDO(64, n_classes, pMCDO=dropoutP, relu=True),
            }

        self.temperatureScaling = temperatureScaling

        if temperatureScaling:
            self.temperature = torch.nn.Parameter(torch.ones(1))

        self.softmaxMCDO = torch.nn.Softmax(dim=1)

        if freeze_seg:
            for layer in self.layers.values():
                for param in layer.parameters():
                    param.requires_grad = False

        if temperatureScaling and freeze_temp:
            self.temperature.requires_grad = False

        for k, v in self.layers.items():
            setattr(self, k, v)

        self.scale_logits = self._get_scale_module(scaling_module)

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.weight.size() == l2.weight.size() and l1.bias.size() == l2.bias.size():
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
                else:

                    num_orig = int(l1.weight.size()[1])
                    num_tiles = int(l2.weight.size()[1]) // int(l1.weight.size()[1])

                    for i in range(num_tiles):
                        l2.weight.data[:, i * num_orig:(i + 1) * num_orig, :, :] = l1.weight.data

        l2.bias.data = l1.bias.data

    def forward(self, inputs, mcdo=True):

        if self.freeze_seg:
            self.eval()

        if self.full_mcdo:
            down1, indices_1, unpool_shape1 = self.layers["down1"](inputs, MCDO=mcdo)
            down2, indices_2, unpool_shape2 = self.layers["down2"](down1, MCDO=mcdo)
        else:
            down1, indices_1, unpool_shape1 = self.layers["down1"](inputs)
            down2, indices_2, unpool_shape2 = self.layers["down2"](down1)

        down3, indices_3, unpool_shape3 = self.layers["down3"](down2, MCDO=mcdo)
        down4, indices_4, unpool_shape4 = self.layers["down4"](down3, MCDO=mcdo)
        down5, indices_5, unpool_shape5 = self.layers["down5"](down4, MCDO=mcdo)

        up5 = self.layers["up5"](down5, indices_5, unpool_shape5, MCDO=mcdo)
        up4 = self.layers["up4"](up5, indices_4, unpool_shape4, MCDO=mcdo)
        up3 = self.layers["up3"](up4, indices_3, unpool_shape3, MCDO=mcdo)

        if self.full_mcdo:
            up2 = self.layers["up2"](up3, indices_2, unpool_shape2, MCDO=mcdo)
            up1 = self.layers["up1"](up2, indices_1, unpool_shape1, MCDO=mcdo)
        else:
            up2 = self.layers["up2"](up3, indices_2, unpool_shape2)
            up1 = self.layers["up1"](up2, indices_1, unpool_shape1)

        if self.temperatureScaling:
            up1 = up1 / self.temperature

        # for param in self.parameters():
        #     print(param.data)

        return up1

    def forwardAvg(self, inputs):

        for i in range(self.mcdo_passes):
            if i == 0:
                x = self.forward(inputs)
            else:
                x = x + self.forward(inputs)

        x = x / self.mcdo_passes
        return x


    def forwardMCDO(self, inputs, mcdo=True):
        with torch.no_grad():
            for i in range(self.mcdo_passes):
                if i == 0:
                    x = self.forward(inputs,mcdo=mcdo).unsqueeze(-1)
                else:
                    x = torch.cat((x, self.forward(inputs).unsqueeze(-1)), -1)

        mean = x.mean(-1)
        variance = x.var(-1)

        prob = self.softmaxMCDO(x)
        entropy, mutual_info = mutualinfo_entropy(prob)  # (batch,512,512)
        
        if self.scale_logits is not None:
            mean = self.scale_logits(mean, variance, mutual_info, entropy)
            
        return mean, variance, entropy, mutual_info

    def forwardMCDO_logits(self, inputs, mcdo=True):   
        with torch.no_grad():
            for i in range(self.mcdo_passes):
                if i == 0:
                    x = self.forward(inputs,mcdo=mcdo).unsqueeze(-1)
                else:
                    x = torch.cat((x, self.forward(inputs).unsqueeze(-1)), -1)
        return x

    def _get_scale_module(self, name, n_classes=11, bias_init=None):

        name = str(name)

        return {
            "temperature": TemperatureScaling(n_classes, bias_init),
            "uncertainty": UncertaintyScaling(n_classes, bias_init),
            "LocalUncertaintyScaling": LocalUncertaintyScaling(n_classes, bias_init),
            "GlobalUncertainty": GlobalUncertaintyScaling(n_classes, bias_init),
            "GlobalLocalUncertainty": GlobalLocalUncertaintyScaling(n_classes, bias_init),
            "GlobalEntropyScaling" : GlobalEntropyScaling( n_classes=11,modality=self.modality,isSpatialTemp=False,bias_init=bias_init),
            "None": None
        }[name]
