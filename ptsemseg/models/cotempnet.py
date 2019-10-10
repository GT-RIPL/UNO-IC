import torch.nn as nn
from torch.autograd import Variable

from .fusion.fusion import *
from ptsemseg.models.segnet_mcdo import *
from ptsemseg.utils import mutualinfo_entropy, plotEverything, plotPrediction

class _tempnet(nn.Module):
  def __init__(self,in_channels=3): 
        super(_tempnet, self).__init__()
        self.temp_down1 =  segnetDown2(in_channels, 64)
        self.temp_down2 = segnetDown2(64, 128)
        self.temp_up2 = segnetUp2(128, 64)
        self.temp_up1 = segnetUp2(64, 1)

  def forward(self,inputs):
       
        tdown1, tindices_1, tunpool_shape1 = self.temp_down1(inputs)
        tdown2, tindices_2, tunpool_shape2 = self.temp_down2(tdown1)
        tup2 = self.temp_up2(tdown2, tindices_2, tunpool_shape2)
        tup1 = self.temp_up1(tup2, tindices_1, tunpool_shape1) #[batch,1,512,512]
        temp = tup1.mean((2,3)).unsqueeze(-1).unsqueeze(-1) #(batch,1,1,1)
        tup1 = tup1.masked_fill(tup1 < 0.3, 0.3)
        return  tup1.squeeze(1),temp.view(-1)

class _compnet(nn.Module):
  def __init__(self,in_channels=6): 
        super(_compnet, self).__init__()
        self.temp_down1 =  segnetDown2(in_channels, 64)
        self.temp_down2 = segnetDown2(64, 128)
        self.temp_up2 = segnetUp2(128, 64)
        self.temp_up1 = segnetUp2(64, 1)

  def forward(self,inputs,inputs2):
        inputs2 = torch.cat((inputs,inputs2),1)
        tdown1, tindices_1, tunpool_shape1 = self.temp_down1(inputs2)
        tdown2, tindices_2, tunpool_shape2 = self.temp_down2(tdown1)
        tup2 = self.temp_up2(tdown2, tindices_2, tunpool_shape2)
        tup1 = self.temp_up1(tup2, tindices_1, tunpool_shape1) #[batch,1,512,512]
        temp = tup1.mean((2,3)).unsqueeze(-1).unsqueeze(-1) #(batch,1,1,1)
        tup1 = tup1.masked_fill(tup1 < 0.3, 0.3)
        return  tup1.squeeze(1),temp.view(-1)


class CoTempNet(nn.Module):
    def __init__(self,
                 modality = 'rgb',
                 n_classes=21,
                 in_channels=3,
                 mcdo_passes=1,
                 full_mcdo=False,
                 freeze_seg=True,
                 freeze_temp=True,
                 dropoutP = 0,
                 scaling_module='None',
                 pretrained_rgb=None,
                 pretrained_d=None
                 ):
        super(CoTempNet, self).__init__()


        self.modality = modality

        self.segnet = segnet_mcdo(modality = self.modality,
                                  n_classes=n_classes,
                                  mcdo_passes=mcdo_passes,
                                  dropoutP=dropoutP,
                                  full_mcdo=full_mcdo,
                                  in_channels=in_channels,
                                  temperatureScaling=False,
                                  freeze_seg=freeze_seg,
                                  freeze_temp=freeze_temp, )

        # initialize temp net
        
        self.tempnet = _tempnet()
        #self.compnet = _compnet()

        self.segnet = torch.nn.DataParallel(self.segnet, device_ids=range(torch.cuda.device_count()))

        if self.modality == 'rgb':
            #self.modality = "rgb"
            self.loadModel(self.segnet, pretrained_rgb)

        elif self.modality == 'd':
            #self.modality = "d"
            self.loadModel(self.segnet, pretrained_d)

        #else:
        #    print("no pretrained given")

        # freeze segnet networks
        for param in self.segnet.parameters():
            param.requires_grad = False

        self.softmaxMCDO = torch.nn.Softmax(dim=1)
        self.scale_logits = self._get_scale_module(scaling_module)


    def forward(self,inputs,inputs2,scaling_metrics="softmax entropy"):

        # Freeze batchnorm
        self.segnet.eval()

        # computer logits and uncertainty measures
        seg = self.segnet.module.forwardMCDO_logits(inputs) #(batch,11,512,512,passes)

        temp_map, temp = self.tempnet(inputs) #(batch,512,512)
        #comp_map, comp = self.compnet(inputs,inputs2) #(batch,512,512)
        #tup1 = tup1.masked_fill(tup1 > 1, 1)
        #x = seg #* torch.min(temp_map,comp_map).unsqueeze(-1) #* tup1.unsqueeze(-1)
        mean = seg.mean(-1) #[batch,classes,512,512]
    
        prob = self.softmaxMCDO(seg) #[batch,classes,512,512]
        prob = prob.masked_fill(prob < 1e-9, 1e-9)
        entropy,mutual_info = mutualinfo_entropy(prob)#(batch,512,512)

        #prob_spatial = self.softmaxMCDO(x) #[batch,classes,512,512]
        #prob_spatial = prob_spatial.masked_fill(prob_spatial < 1e-9, 1e-9)
        #entropy_spatial,mutual_info_spatial = mutualinfo_entropy(prob_spatial)#(batch,512,512)

        #import ipdb;ipdb.set_trace()
        #import ipdb;ipdb.set_trace()
        if self.scale_logits != None:
          DR = self.scale_logits(entropy,mutual_info, temp1=temp_map,mode=scaling_metrics) #(batch,1,1,1)
          #mean_comp = mean * torch.min(DR,comp_map.unsqueeze(1))
          mean = mean * DR
          #import ipdb;ipdb.set_trace() 
        else:
          DR = 0
        return mean, entropy, mutual_info,temp_map,temp,entropy.mean((1,2)),mutual_info.mean((1,2)),DR

    def loadModel(self, model, path):
        model_pkl = path

        print(path)
        if os.path.isfile(model_pkl):
            pretrained_dict = torch.load(model_pkl)['model_state']
            model_dict = model.state_dict()
            #import ipdb;ipdb.set_trace()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (
                    k in model_dict)}  # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}
            print("Loaded {} pretrained parameters".format(len(pretrained_dict)))
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            #import ipdb;ipdb.set_trace()
            # 3. load the new state dict
            model.load_state_dict(pretrained_dict)
            #import ipdb;ipdb.set_trace()
        else:
            print("no pretrained given")
            #exit()

    def _get_scale_module(self, name, n_classes=11, bias_init=None):

        name = str(name)

        return {
            "temperature": TemperatureScaling(n_classes, bias_init),
            "uncertainty": UncertaintyScaling(n_classes, bias_init),
            "LocalUncertaintyScaling": LocalUncertaintyScaling(n_classes, bias_init),
            "GlobalUncertainty": GlobalUncertaintyScaling(n_classes, bias_init),
            "GlobalLocalUncertainty": GlobalLocalUncertaintyScaling(n_classes, bias_init),
            "GlobalScaling" : GlobalScaling(modality=self.modality),
            "None": None
        }[name]
 
