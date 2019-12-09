import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from ..segnet import segnet
from .deeplab import DeepLab
from .decoder import build_decoder
from .fusion import *
from ptsemseg.utils import mutualinfo_entropy, plotEverything, plotPrediction
from ptsemseg.models.utils import *
import torchvision.models as models

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

class SSMA(nn.Module):
    def __init__(self, backbone='segnet', output_stride=16, n_classes=11,
                 sync_bn=True, freeze_bn=False):
        super(SSMA, self).__init__()
        if backbone == 'segnet':
            self.expert_A = segnet(n_classes=n_classes, in_channels=3, is_unpooling=True)
            self.expert_B = segnet(n_classes=n_classes, in_channels=3, is_unpooling=True)
            vgg16 = models.vgg16(pretrained=True)
            self.expert_A.init_vgg16_params(vgg16)
            self.expert_B.init_vgg16_params(vgg16)
            self.SSMA_skip1 = _SSMABlock(24, 4)
            self.SSMA_skip2 = _SSMABlock(24, 4)
            self.SSMA_ASPP = _SSMABlock(512, 4)
        else:
            self.expert_A = DeepLab(backbone, output_stride, n_classes, sync_bn, freeze_bn)
            self.expert_B = DeepLab(backbone, output_stride, n_classes, sync_bn, freeze_bn)
            self.SSMA_skip1 = _SSMABlock(64, 4)
            self.SSMA_skip2 = _SSMABlock(512, 4)
            self.SSMA_ASPP = _SSMABlock(512, 4)
        self.modality = 'rgbd'
        self.decoder = _Decoder(n_classes, in_channels=512)
        self.softmaxMCDO = torch.nn.Softmax(dim=1)
        # self.tempnet_rgb = _tempnet()
        # self.tempnet_d = _tempnet()

    def forward(self, input):
        # print(input.shape)
        # import pdb;pdb.set_trace()
        A, A_llf1, A_llf2, A_aspp = self.expert_A.forward_SSMA(input[:, :3, :, :])
        B, B_llf1, B_llf2, B_aspp = self.expert_B.forward_SSMA(input[:, 3:, :, :])
        # import pdb;pdb.set_trace()
        fused_ASPP = self.SSMA_ASPP(A_aspp, B_aspp,DR)
        fused_skip1 = self.SSMA_skip1(A_llf1, B_llf1,DR)
        fused_skip2 = self.SSMA_skip2(A_llf2, B_llf2,DR)
        x = self.decoder(fused_ASPP, fused_skip1, fused_skip2)
        
        prob = self.softmaxMCDO(x.unsqueeze(-1)) #[batch,classes,512,512]
        prob = prob.masked_fill(prob < 1e-9, 1e-9)
        entropy,mutual_info = mutualinfo_entropy(prob)#(batch,512,512)


        #temp_map_A, _ = self.tempnet_rgb(input[:, :3, :, :])
        #temp_map_B, _ = self.tempnet_d(input[:, 3:, :, :])

        #if self.scale_logits != None:
          #DR = self.scale_logits(entropy,mutual_info,mode=scaling_metrics) #(batch,1,1,1)
          #mean_comp = mean * torch.min(DR,comp_map.unsqueeze(1))
          #x = x * DR
          #import ipdb;ipdb.set_trace() 
        #else:
          #DR = 0

        return x, entropy, mutual_info



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


class _SSMABlock(nn.Module):
    def __init__(self, n_channels, compression_rate=6):
        super(_SSMABlock, self).__init__()
        self.bottleneck = nn.Conv2d(
            2 * n_channels,
            n_channels // compression_rate,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )
        self.gate = nn.Conv2d(
            n_channels // compression_rate,
            2 * n_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )
        conv_mod = nn.Conv2d(
            2 * n_channels,
            n_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )
        self.fuser = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_channels)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, B, DR = 0):
        #DR (batch,1,1,1)
        if DR != 0:
            AB = torch.cat([A*DR[0], B*DR[1]], dim=1)
        else:
            AB = torch.cat([A, B], dim=1)
        G = self.bottleneck(AB)
        G = F.relu(G)
        G = self.gate(G)
        G = self.sigmoid(G)
        #if DR != 0:
            #if DR[0] == 1 and DR[1] == 1:
            #    AB = AB * G
            #else:
        #    AB = torch.cat([A*DR[0], B*DR[1]], dim=1) * G
        #else:
        AB = AB * G
        fused = self.fuser(AB)
        return fused

class _Decoder(nn.Module):
    def __init__(self, n_classes=11, in_channels=512, compress_size=24):
        super(_Decoder, self).__init__()
        
        self.leg1 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(in_channels))
                                  
        self.compress1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       conv2DBatchNormRelu(in_channels,compress_size,1,1,0))
                                       
        self.leg2 = nn.Sequential(conv2DBatchNormRelu(in_channels + compress_size, in_channels, 3, 1, 1),
                                  conv2DBatchNormRelu(in_channels, in_channels, 3, 1, 1),
                                  nn.ConvTranspose2d(in_channels, in_channels, 8, stride=4, padding=2),
                                  nn.BatchNorm2d(in_channels))
                                  
      
        self.compress2 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       conv2DBatchNormRelu(in_channels,compress_size,1,1,0))


        self.leg3 = nn.Sequential(conv2DBatchNormRelu(in_channels + compress_size,in_channels,3,1,1),
                                  conv2DBatchNormRelu(in_channels,in_channels,3,1,1),
                                  conv2DBatchNormRelu(in_channels,in_channels,3,1,1),
                                  nn.ConvTranspose2d(in_channels, n_classes, 8, stride=4, padding=2),
                                  nn.BatchNorm2d(n_classes))
        
    def forward(self, ASSP, skip1, skip2):
        x = self.leg1(ASSP)
        features1 = self.compress1(x) * skip2
        x = torch.cat([x, features1], dim=1)
        x = self.leg2(x)
        features2 = self.compress2(x) * skip1
        x = torch.cat([x, features2], dim=1)
        x = self.leg3(x)
        return x

if __name__ == "__main__":
    model = SSMA(backbone='mobilenet', output_stride=16, n_classes=11)
    model.eval()
    input = torch.rand(2, 6, 512, 512)
    output = model(input)
    print(output.size())