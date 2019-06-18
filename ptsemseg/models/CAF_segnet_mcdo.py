from ptsemseg.models.segnet import *


class fused_segnet(nn.Module):
    def __init__(self,
                 n_classes=21,
                 in_channels=3,
                 is_unpooling=True
                 ):
        super(fused_segnet, self).__init__()

        self.rgb_segnet = segnet(n_classes, in_channels, is_unpooling)
        self.d_segnet = segnet(n_classes, in_channels, is_unpooling)

        self.gatedFusion = GatedFusion(n_classes)

    def forward(self, inputs):
        inputs_rgb = inputs[:, :3, :, :]
        inputs_d = inputs[:, 3:, :, :]
        rgb = self.rgb_segnet(inputs_rgb)
        d = self.d_segnet(inputs_d)

        x = self.ConditionalAttentionFusion(rgb, d)

        return x
