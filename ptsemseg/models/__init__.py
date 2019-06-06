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

def get_model(model_dict, 
              n_classes, 
              input_size=(512,512),
              batch_size=2,
              mcdo_passes=1,
              fixed_mcdo=False,
              dropoutP=0.1,
              in_channels=3,
              start_layer="convbnrelu1_1",
              end_layer="classification",
              learned_uncertainty="none",
              version=None, 
              reduction=1.0,
              recalibrator=None,
              temperatureScaling=False,
              bins=0,
              device="cpu"):
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet_mcdo":
        model = model(n_classes=n_classes, 
                      input_size=input_size,
                      batch_size=batch_size,
                      version=version, 
                      reduction=reduction, 
                      mcdo_passes=mcdo_passes,
                      fixed_mcdo=fixed_mcdo,
                      dropoutP=dropoutP,
                      learned_uncertainty=learned_uncertainty,
                      in_channels=in_channels,
                      start_layer=start_layer,
                      end_layer=end_layer,
                      device=device,
                      recalibrator=recalibrator,
                      temperatureScaling=temperatureScaling,
                      bins=bins,
                      **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

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
                      end_layer=end_layer,
                      **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

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
        }[name]
    except:
        raise("Model {} not available".format(name))
