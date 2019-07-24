
class GatedFusion(nn.Module):
    def __init__(self, n_classes):
        super(GatedFusion, self).__init__()

        self.conv = nn.Conv2d(
            2 * n_classes,
            n_classes,
            1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, d):

        fusion = torch.cat([rgb, d], dim=1)

        G = self.conv(fusion)
        G = self.sigmoid(G)

        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = P_rgb + P_d

        return P_fusion

class ConditionalAttentionFusion(nn.Module):
    def __init__(self, n_classes):
        super(ConditionalAttentionFusion, self).__init__()
        self.bottleneck = nn.Conv2d(
            4 * n_channels,
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
        self.n_classes = n_classes

    def forward(self, rgb, d, rgb_var, d_var):

        ABCD = torch.cat([rgb, d, rgb_var, d_var], dim=1)
        
        G = self.bottleneck(ABCD)
        G = F.relu(G)
        G = self.gate(G)
        G = self.sigmoid(G)

        AB = AB * G
        
        fused = self.fuser(ABCD)

        return fused

class UncertaintyGatedFusion(nn.Module):
    def __init__(self, n_classes):
        super(UncertaintyGatedFusion, self).__init__()
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
        self.n_classes = n_classes

    def forward(self, rgb, d, rgb_var, d_var):


        AB = torch.cat([rgb, d], dim=1)
        CD = torch.cat([rgb_var, d_var], dim=1)
       
        G = self.bottleneck(CD)
        G = F.relu(G)
        G = self.gate(G)
        G = self.sigmoid(G)

        AB = AB * G
        
        fused = self.fuser(fusion)

        return fused
        