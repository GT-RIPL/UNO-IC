from .resnet import *   
from .inception import *

# def get_model(name,**kwargs):
    # model = _get_model_instance(name)
    # return model()
    #return model


def get_model(name,**kwargs):
    # import ipdb;ipdb.set_trace()
    # return resnet32(num_classes=kwargs['num_classes'],use_norm=kwargs['use_norm'])
    if name == "resnet152":
        return resnet152(num_classes=kwargs['num_classes'])
    elif name == "resnet50":
        return resnet50(num_classes=kwargs['num_classes'])
    elif name == "inceptionV3":
        return inception_v3(num_classes=kwargs['num_classes'])
    else: 
        raise ValueError("Model {} not available".format(name))
    print('adfasd')