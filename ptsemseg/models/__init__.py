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
from ptsemseg.models.fusion.deeplab import DeepLab
from ptsemseg.models.fusion.CAFnet import CAFnet
from ptsemseg.models.tempnet import tempnet


def get_model(name,
              n_classes,
              input_size=(512, 512),
              batch_size=2,
              mcdo_passes=1,
              fixed_mcdo=False,
              dropoutP=0.1,
              full_mcdo=False,
              in_channels=3,
              start_layer="convbnrelu1_1",
              end_layer="classification",
              learned_uncertainty="none",
              version=None,
              reduction=1.0,
              recalibration=None,
              recalibrator=None,
              temperatureScaling=False,
              freeze_seg=False,
              freeze_temp=False,
              fusion_module="1.1",
              scaling_module=None,
              bins=0,
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
                      input_size=input_size,
                      batch_size=batch_size,
                      version=version,
                      reduction=reduction,
                      mcdo_passes=mcdo_passes,
                      dropoutP=dropoutP,
                      full_mcdo=full_mcdo,
                      in_channels=in_channels,
                      start_layer=start_layer,
                      end_layer=end_layer,
                      device=device,
                      recalibration=recalibration,
                      recalibrator=recalibrator,
                      temperatureScaling=temperatureScaling,
                      freeze_seg=freeze_seg,
                      freeze_temp=freeze_temp,
                      bins=bins)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "tempnet":
        model = model(n_classes=n_classes,
                      input_size=input_size,
                      batch_size=batch_size,
                      version=version,
                      reduction=reduction,
                      mcdo_passes=mcdo_passes,
                      dropoutP=dropoutP,
                      full_mcdo=full_mcdo,
                      in_channels=in_channels,
                      start_layer=start_layer,
                      end_layer=end_layer,
                      device=device,
                      recalibration=recalibration,
                      recalibrator=recalibrator,
                      freeze_seg=freeze_seg,
                      freeze_temp=freeze_temp,
                      bins=bins)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "CAFnet" or name == "CAF_segnet":
        model = model(backbone="segnet",
                      n_classes=n_classes,
                      input_size=input_size,
                      batch_size=batch_size,
                      version=version,
                      reduction=reduction,
                      mcdo_passes=mcdo_passes,
                      dropoutP=dropoutP,
                      full_mcdo=full_mcdo,
                      in_channels=in_channels,
                      start_layer=start_layer,
                      end_layer=end_layer,
                      device=device,
                      recalibration=recalibration,
                      recalibrator=recalibrator,
                      temperatureScaling=temperatureScaling,
                      bins=bins,
                      pretrained_rgb=pretrained_rgb,
                      pretrained_d=pretrained_d,
                      fusion_module=fusion_module,
                      scaling_module=scaling_module)

    elif name == "fused_segnet":
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.rgb_segnet.init_vgg16_params(vgg16)
        model.d_segnet.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes)

    elif name == "pspnet":
        model = model(n_classes=n_classes,
                      input_size=input_size,
                      version=version,
                      reduction=reduction,
                      mcdo_passes=mcdo_passes,
                      dropoutP=dropoutP,
                      learned_uncertainty=learned_uncertainty,
                      in_channels=in_channels,
                      start_layer=start_layer,
                      end_layer=end_layer)

    elif name == "SSMA":
        model = model(backbone='resnet', output_stride=16, num_classes=n_classes, sync_bn=True, freeze_bn=False)
    elif name == "DeepLab":
        model = model(backbone='resnet', output_stride=16, num_classes=n_classes, sync_bn=True, freeze_bn=False)
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
            "DeepLab": DeepLab,
            "tempnet": tempnet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
