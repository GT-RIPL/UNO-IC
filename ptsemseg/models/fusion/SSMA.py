import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from deeplab import DeepLab
from decoder import build_decoder

class SSMA(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
                 
                 
        super(SSMA, self).__init__()
        self.expert_A = DeepLab(backbone, output_stride, num_classes, sync_bn, freeze_bn)
        self.expert_B = DeepLab(backbone, output_stride, num_classes, sync_bn, freeze_bn)
     
        self.SSMA_skip1 = _SSMABlock(24, 4)
        self.SSMA_ASPP = _SSMABlock(256, 4)
     
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            model1.freeze_bn()
            model2.freeze_bn()

    def forward(self, input):
        
        A, A_aspp, A_llf = self.expert_A.forward_SSMA(input[:, :3, :, :])
        B, B_aspp, B_llf = self.expert_B.forward_SSMA(input[:, 3:, :, :])
        
        fused_skip = self.SSMA_skip1(A_llf, B_llf)
        fused_ASPP = self.SSMA_ASPP(A_aspp, A_aspp)
        
        x = self.decoder(fused_ASPP, fused_skip)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # TODO add fusion between expert A, B, and AB

        return x

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

    def forward(self, A, B):
        AB = torch.cat([A, B], dim=1)
        
        G = self.bottleneck(AB)
        G = F.relu(G)
        G = self.gate(G)
        G = self.sigmoid(G)

        AB = AB * G
        
        fused = self.fuser(AB)

        return fused


if __name__ == "__main__":
    model = SSMA(backbone='mobilenet', output_stride=16, num_classes=11)
    model.eval()
    input = torch.rand(2, 6, 512, 512)
    output = model(input)
    print(output.size())


