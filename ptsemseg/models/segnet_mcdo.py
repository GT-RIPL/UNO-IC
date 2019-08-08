import torch.nn as nn
from torch.autograd import Variable

from ptsemseg.models.recalibrator import *
from ptsemseg.utils import save_pred


class segnet_mcdo(nn.Module):
    def __init__(self,
                 n_classes=21,
                 in_channels=3,
                 is_unpooling=True,
                 input_size=(473, 473),
                 batch_size=2,
                 version=None,
                 mcdo_passes=1,
                 dropoutP=0.1,
                 full_mcdo=False,
                 start_layer="down1",
                 end_layer="up1",
                 reduction=1.0,
                 device="cpu",
                 recalibrator="None",
                 temperatureScaling=False,
                 freeze=False,
                 bins=0
                 ):
        super(segnet_mcdo, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.mcdo_passes = mcdo_passes
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.dropoutP = dropoutP
        self.full_mcdo = full_mcdo
        self.device = device

        # Select Recalibrator
        self.temperatureScaling = temperatureScaling
        self.recalibrator = recalibrator

        if recalibrator != "None" and bins > 0:
            self.ranges = list(zip([1. * a / bins for a in range(bins + 2)][:-2],
                                   [1. * a / bins for a in range(bins + 2)][1:]))
            if recalibrator == "HistogramFlat":
                self.calibrationPerClass = [HistogramFlatRecalibrator(n, self.ranges, device) for n in
                                            range(self.n_classes)]
            elif recalibrator == "HistogramLinear":
                self.calibrationPerClass = [HistogramLinearRecalibrator(n, self.ranges, device) for n in
                                            range(self.n_classes)]
            elif "Polynomial" in recalibrator:
                degree = int(recalibrator.split("_")[-1])
                self.calibrationPerClass = [PolynomialRecalibrator(n, self.ranges, degree, device) for n in
                                            range(self.n_classes)]
            elif "Isotonic" in recalibrator:
                self.calibrationPerClass = [IsotonicRecalibrator(n, device) for n in range(self.n_classes)]
            elif "Platt" in recalibrator:
                self.calibrationPerClass = [PlattRecalibrator(n, device) for n in range(self.n_classes)]
            else:
                print("Recalibrator: Not Supported")
                exit()

        if not self.full_mcdo:

            self.layers = {
                "down1": segnetDown2(self.in_channels, 64),
                "down2": segnetDown2(64, 128),
                "down3": segnetDown3MCDO(128, 256, pMCDO=dropoutP),
                "down4": segnetDown3MCDO(256, 512, pMCDO=dropoutP),
                "down5": segnetDown3MCDO(512, 512, pMCDO=dropoutP),
                "up5": segnetUp3MCDO(512, 512, pMCDO=dropoutP),
                "up4": segnetUp3MCDO(512, 256, pMCDO=dropoutP),
                "up3": segnetUp3MCDO(256, 128, pMCDO=dropoutP),
                "up2": segnetUp2(128, 64),
                "up1": segnetUp2(64, n_classes, relu=True),
            }

        else:

            self.layers = {
                "down1": segnetDown2MCDO(self.in_channels, 64, pMCDO=dropoutP),
                "down2": segnetDown2MCDO(64, 128, pMCDO=dropoutP),
                "down3": segnetDown3MCDO(128, 256, pMCDO=dropoutP),
                "down4": segnetDown3MCDO(256, 512, pMCDO=dropoutP),
                "down5": segnetDown3MCDO(512, 512, pMCDO=dropoutP),
                "up5": segnetUp3MCDO(512, 512, pMCDO=dropoutP),
                "up4": segnetUp3MCDO(512, 256, pMCDO=dropoutP),
                "up3": segnetUp3MCDO(256, 128, pMCDO=dropoutP),
                "up2": segnetUp2MCDO(128, 64, pMCDO=dropoutP),
                "up1": segnetUp2MCDO(64, n_classes, pMCDO=dropoutP, relu=True),
            }

        if self.temperatureScaling:
            self.temperature = torch.nn.Parameter(torch.ones(1))

        if freeze:
            for layer in self.layers.values():
                for param in layer.parameters():
                    param.requires_grad = False

        self.softmaxMCDO = torch.nn.Softmax(dim=1)

        for k, v in self.layers.items():
            setattr(self, k, v)

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.weight.size() == l2.weight.size() and l1.bias.size() == l2.bias.size():
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
                else:

                    num_orig = int(l1.weight.size()[1])
                    num_tiles = int(l2.weight.size()[1]) // int(l1.weight.size()[1])

                    for i in range(num_tiles):
                        l2.weight.data[:, i * num_orig:(i + 1) * num_orig, :, :] = l1.weight.data

        l2.bias.data = l1.bias.data

    def forward(self, inputs, mcdo=True):

        if self.full_mcdo:
            down1, indices_1, unpool_shape1 = self.layers["down1"](inputs, MCDO=mcdo)
            down2, indices_2, unpool_shape2 = self.layers["down2"](down1, MCDO=mcdo)
        else:
            down1, indices_1, unpool_shape1 = self.layers["down1"](inputs)
            down2, indices_2, unpool_shape2 = self.layers["down2"](down1)

        down3, indices_3, unpool_shape3 = self.layers["down3"](down2, MCDO=mcdo)
        down4, indices_4, unpool_shape4 = self.layers["down4"](down3, MCDO=mcdo)
        down5, indices_5, unpool_shape5 = self.layers["down5"](down4, MCDO=mcdo)

        up5 = self.layers["up5"](down5, indices_5, unpool_shape5, MCDO=mcdo)
        up4 = self.layers["up4"](up5, indices_4, unpool_shape4, MCDO=mcdo)
        up3 = self.layers["up3"](up4, indices_3, unpool_shape3, MCDO=mcdo)

        if self.full_mcdo:
            up2 = self.layers["up2"](up3, indices_2, unpool_shape2, MCDO=mcdo)
            up1 = self.layers["up1"](up2, indices_1, unpool_shape1, MCDO=mcdo)
        else:
            up2 = self.layers["up2"](up3, indices_2, unpool_shape2)
            up1 = self.layers["up1"](up2, indices_1, unpool_shape1)

        if self.temperatureScaling:
            up1 = up1 / (self.temperature + 1e-5)

        return up1

    def forwardAvg(self, inputs):

        for i in range(self.mcdo_passes):
            if i == 0:
                x = self.forward(inputs)
            else:
                x = x + self.forward(inputs)

        x = x / self.mcdo_passes
        return x

    def forwardMCDO(self, inputs, recalType="None", softmax=False):

        for i in range(self.mcdo_passes):
            if i == 0:
                x_bp = self.forward(inputs)
                x = x_bp.unsqueeze(-1)
            else:
                x = torch.cat((x, self.forward(inputs).unsqueeze(-1)), -1)

        mean = x.mean(-1)
        variance = x.std(-1)

        # Uncalibrated Softmax Mean and Variance
        if recalType != "None":

            mean = self.softmaxMCDO(x).mean(-1)
            variance = self.softmaxMCDO(x).std(-1)

            if recalType == "beforeMCDO":
                for c in range(self.n_classes):
                    x[:, c, :, :, :] = self.calibrationPerClass[c].predict(x[:, c, :, :, :].reshape(-1)).reshape(
                        x[:, c, :, :, :].shape)
                mean = self.softmaxMCDO(x).mean(-1)
                variance = self.softmaxMCDO(x).std(-1)
            elif recalType == "afterMCDO":
                for c in range(self.n_classes):
                    mean[:, c, :, :] = self.calibrationPerClass[c].predict(mean[:, c, :, :].reshape(-1)).reshape(
                        mean[:, c, :, :].shape)

        prob = self.softmaxMCDO(x)
        # entropy = predictive_entropy(prob)
        # mutual_info = mutul_information(prob)
        entropy, mutual_info = mutualinfo_entropy(prob)  # (2,512,512)
        return mean, variance, entropy, mutual_info

    def applyCalibration(self, output):

        for c in range(self.n_classes):
            output[:, c, :, :] = self.calibrationPerClass[c].predict(output[:, c, :, :].reshape(-1)).reshape(
                output[:, c, :, :].shape)

        return output

    def showCalibration(self, output, label, logdir, model, iteration):

        recal_output = self.applyCalibration(output.clone())

        ###########
        # Overall #
        ###########
        fig, axes = plt.subplots(1, 2)

        # Plot Predicted Variance Against Observed/Empirical Variance
        x, y = calcStatistics(output, label, self.ranges)

        x = list(x)
        y = list(y)

        axes[0].plot(x, y, '.')
        axes[0].set_title("Uncalibrated")
        axes[0].set_xlabel("uncalibrated confidence")
        axes[0].set_ylabel("emperical probability")

        # Convert Predicted Variances to Calibrated Variances
        x, y = calcStatistics(recal_output, label, self.ranges)
        x = list(x)
        y = list(y)

        axes[1].plot(x, y)
        axes[1].set_title("Recalibrated")
        axes[1].set_xlabel("calibrated confidence")
        axes[1].set_ylabel("emperical probability")

        # calculating expected calibration error
        ECE = sum([abs(i - j) for i, j, in zip(x, y)]) / len(x)
        fig.suptitle('Expected Calibration Error: {}'.format(ECE), fontsize=16)

        path = "{}/{}/{}".format(logdir, 'calibration', model)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.tight_layout()
        plt.savefig("{}/calibratedOverall{}.png".format(path, iteration))
        plt.close(fig)

        ############################
        # All Classes Uncalibrated #
        ############################
        fig, axes = plt.subplots(3, self.n_classes // 3 + 1)

        for c in range(self.n_classes):
            x, y = calcClassStatistics(output, label, self.ranges, c)
            x = list(x)
            y = list(y)
            axes[(c + 1) // (self.n_classes // 3 + 1), (c + 1) % (self.n_classes // 3 + 1)].plot(x, y)
            axes[(c + 1) // (self.n_classes // 3 + 1), (c + 1) % (self.n_classes // 3 + 1)].set_title(
                "Class: {}".format(c))

        path = "{}/{}/{}".format(logdir, 'calibration', model)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.tight_layout()
        plt.savefig("{}/uncalibratedPerClass{}.png".format(path, iteration))
        plt.close(fig)

        ##########################
        # All Classes Calibrated #
        ##########################
        fig, axes = plt.subplots(3, self.n_classes // 3 + 1)

        for c in range(self.n_classes):
            x, y = calcClassStatistics(recal_output, label, self.ranges, c)
            x = list(x)
            y = list(y)

            axes[(c + 1) // (self.n_classes // 3 + 1), (c + 1) % (self.n_classes // 3 + 1)].plot(x, y)
            axes[(c + 1) // (self.n_classes // 3 + 1), (c + 1) % (self.n_classes // 3 + 1)].set_title(
                "Class: {}".format(c))

        path = "{}/{}/{}".format(logdir, 'calibration', model)
        if not os.path.exists(path):
            os.makedirs(path)

        plt.tight_layout()
        plt.savefig("{}/calibratedPerClass{}.png".format(path, iteration))
        plt.close(fig)

        torch.cuda.empty_cache()
