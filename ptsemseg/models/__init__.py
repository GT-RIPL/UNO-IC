import copy
import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.segnet_mcdo import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.icnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.frrn import *
from ptsemseg.models.fusion.SSMA import SSMA
from ptsemseg.models.fusion.shared_SSMA import shared_SSMA
from ptsemseg.models.fusion.deeplab import DeepLab
from ptsemseg.models.fusion.CAFnet import CAFnet
from ptsemseg.models.fusion.fusenet import FuseNet
from ptsemseg.models.tempnet import TempNet
from ptsemseg.models.cotempnet import CoTempNet
from ptsemseg.models.fusion.separate_SSMA import separate_SSMA
from ptsemseg.models.fusion.joint_SSMA import joint_SSMA 
from ptsemseg.models.fusion.robust_SSMA import robust_SSMA 


def get_model(name,
              modality = 'rgb',
              n_classes = 11,
              input_size=(512, 512),
              mcdo_passes=6,
              dropoutP=0.5,
              full_mcdo=False,
              in_channels=3,
              temperatureScaling=False,
              freeze_seg=False,
              freeze_temp=False,
              fusion_module="1.1",
              scaling_module=None,
              pretrained_rgb=None,
              pretrained_d=None,
              device="cpu"):
    model = _get_model_instance(name)

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet_mcdo":
        model = model(n_classes=n_classes,
                      modality = modality,
                      mcdo_passes=mcdo_passes,
                      dropoutP=dropoutP,
                      full_mcdo=full_mcdo,
                      in_channels=in_channels,
                      temperatureScaling=temperatureScaling,
                      scaling_module=scaling_module,
                      freeze_seg=freeze_seg,
                      freeze_temp=freeze_temp)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "tempnet":
        model = model(n_classes=n_classes,
                      modality = modality,
                      mcdo_passes=mcdo_passes,
                      dropoutP=dropoutP,
                      full_mcdo=full_mcdo,
                      in_channels=in_channels,
                      scaling_module=scaling_module,
                      pretrained_rgb=pretrained_rgb,
                      pretrained_d=pretrained_d)
    elif name == "CAFnet" or name == "CAF_segnet":
        model = model(backbone="segnet",
                      n_classes=n_classes,
                      mcdo_passes=mcdo_passes,
                      dropoutP=dropoutP,
                      full_mcdo=full_mcdo,
                      in_channels=in_channels,
                      temperatureScaling=temperatureScaling,
                      pretrained_rgb=pretrained_rgb,
                      pretrained_d=pretrained_d,
                      fusion_module=fusion_module,
                      scaling_module=scaling_module)
    elif name == "fusenet":
        model = model(n_classes, use_class=False)

    elif name == "unet":
        model = model(n_classes=n_classes)

    elif name == "pspnet":
        model = model(n_classes=n_classes,
                      input_size=input_size,
                      mcdo_paSsses=mcdo_passes,
                      dropoutP=dropoutP,
                      in_channels=in_channels, )

    elif name == "SSMA":
        model = model(backbone='segnet', output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    elif name == "shared_SSMA":
        model = model(backbone='segnet', output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    elif name == "joint_SSMA":
        model = model(backbone='segnet', output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    elif name == "separate_SSMA":
        model = model(backbone='segnet', output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    elif name == "robust_SSMA":
        model = model(backbone='segnet', output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    elif name == "DeepLab":
        model = model(backbone='segnet', output_stride=16, n_classes=n_classes, sync_bn=True, freeze_bn=False)
    else:
        model = model(n_classes=n_classes)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,
            "segnet": segnet,
            "segnet_mcdo": segnet_mcdo,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
            "CAFnet": CAFnet,
            "CAF_segnet": CAFnet,
            "SSMA": SSMA,
            "shared_SSMA" : shared_SSMA,
            "joint_SSMA" : joint_SSMA,
            "separate_SSMA" : separate_SSMA,
            "robust_SSMA" : robust_SSMA,
            "DeepLab": DeepLab,
            "fusenet": FuseNet,
            "tempnet": CoTempNet#TempNet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
