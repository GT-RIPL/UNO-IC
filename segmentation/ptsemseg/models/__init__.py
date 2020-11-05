import copy
import torchvision.models as models

from ptsemseg.models.segnet import *
from ptsemseg.models.segnet_mcdo import *
from ptsemseg.models.deeplab import DeepLab
from ptsemseg.models.fusion.fusenet import FuseNet
from ptsemseg.models.fusion.SSMA import SSMA


def get_model(name,
              n_classes = 11,
              input_size=(512, 512),
              mcdo_passes=6,
              dropoutP=0.5,
              full_mcdo=False,
              in_channels=3,
              backbone='segnet',
              device="cpu"):
    model = _get_model_instance(name)

    if name == "segnet":
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet_mcdo":
        model = model(n_classes=n_classes,
                      mcdo_passes=mcdo_passes,
                      dropoutP=dropoutP,
                      full_mcdo=full_mcdo,
                      in_channels=in_channels,)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

   
    elif name == "fusenet":
        model = model(n_classes, use_class=False)
    elif name == "SSMA":
        model = model(backbone=backbone, output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    elif name == "DeepLab":
        model = model(backbone=backbone, output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    else:
        model = model(n_classes=n_classes)

    return model


def _get_model_instance(name):
    try:
        return {
            "segnet": segnet,
            "segnet_mcdo": segnet_mcdo,
            "SSMA": SSMA,
            "DeepLab": DeepLab,
            "fusenet": FuseNet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
