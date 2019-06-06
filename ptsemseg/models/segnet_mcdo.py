import torch.nn as nn
from torch.autograd import Variable

from ptsemseg.models.utils import *
from ptsemseg.models.recalibrator import *


class segnet_mcdo(nn.Module):
    def __init__(self,
                 n_classes=21,
                 in_channels=3,
                 is_unpooling=True,
                 input_size=(473, 473),
                 batch_size=2,
                 version=None,
                 mcdo_passes=1,
                 fixed_mcdo=False,
                 dropoutP=0.1,
                 learned_uncertainty="none",
                 start_layer="down1",
                 end_layer="up1",
                 reduction=1.0,
                 device="cpu",
                 recalibrator="None",
                 temperatureScaling="False",
                 bins=0
                 ):
        super(segnet_mcdo, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.mcdo_passes = mcdo_passes
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.dropoutP = dropoutP
        self.fixed_mcdo = fixed_mcdo
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

        if self.temperatureScaling:

            self.temperature = torch.nn.Parameter(torch.ones(1))

        if not self.fixed_mcdo:
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
                "up1": segnetUp2(64, n_classes),
            }
        else:
            self.layers = {
                "down1": segnetDown2(self.in_channels, 64),
                "down2": segnetDown2(64, 128),
                "down3": segnetDown3MCDO(128, 256),
                "down4": segnetDown3MCDO(256, 512),
                "down5": segnetDown3MCDO(512, 512),
                "up5": segnetUp3MCDO(512, 512),
                "up4": segnetUp3MCDO(512, 256),
                "up3": segnetUp3MCDO(256, 128),
                "up2": segnetUp2(128, 64),
                "up1": segnetUp2(64, n_classes),
            }

        self.dropouts = {k: nn.Dropout2d(p=dropoutP, inplace=False) for k in self.layers.keys()}

        # self.dropout_layers = ["down3","down4","down5","up5","up4","up3"]

        # inp = torch.Tensor(512,512,3)
        # f = mod.forward(autograd.Variable(torch.Tensor(1, *inp.shape)))
        # print( int(np.prod(f.size()[1:])) )

        # inputs torch.Size([2, 3, 512, 512])
        # down1 torch.Size([2, 64, 256, 256])
        # down2 torch.Size([2, 128, 128, 128])
        # down3 torch.Size([2, 256, 64, 64])
        # down4 torch.Size([2, 512, 32, 32])
        # down5 torch.Size([2, 512, 16, 16])
        # up1 torch.Size([2, 11, 512, 512])
        # up2 torch.Size([2, 64, 256, 256])
        # up3 torch.Size([2, 128, 128, 128])
        # up4 torch.Size([2, 256, 64, 64])
        # up5 torch.Size([2, 512, 32, 32])


        if self.fixed_mcdo:
            self.dropout_masks = {p:
                {
                    "down3": Variable((1. / (1 - self.dropoutP)) * torch.bernoulli(
                        (1 - self.dropoutP) * torch.ones(self.batch_size, 256, 64, 64))).to(device),
                    "down4": Variable((1. / (1 - self.dropoutP)) * torch.bernoulli(
                        (1 - self.dropoutP) * torch.ones(self.batch_size, 512, 32, 32))).to(device),
                    "down5": Variable((1. / (1 - self.dropoutP)) * torch.bernoulli(
                        (1 - self.dropoutP) * torch.ones(self.batch_size, 512, 16, 16))).to(device),
                    "up5": Variable((1. / (1 - self.dropoutP)) * torch.bernoulli(
                        (1 - self.dropoutP) * torch.ones(self.batch_size, 512, 32, 32))).to(device),
                    "up4": Variable((1. / (1 - self.dropoutP)) * torch.bernoulli(
                        (1 - self.dropoutP) * torch.ones(self.batch_size, 256, 64, 64))).to(device),
                    "up3": Variable((1. / (1 - self.dropoutP)) * torch.bernoulli(
                        (1 - self.dropoutP) * torch.ones(self.batch_size, 128, 128, 128))).to(device),
                } for p in range(self.mcdo_passes)}

        # print(torch.bernoulli(0.5*torch.ones(10)))
        # print(Variable(torch.bernoulli(0.5*torch.ones(10))))
        # print(self.dropout_masks[0]["down3"][0,0,0,:10])
        # print(self.dropout_masks[1]["down3"][0,0,0,:10])
        # exit()

        self.ordered_layers = [
            "down1",
            "down2",
            "down3",
            "down4",
            "down5",
            "up5",
            "up4",
            "up3",
            "up2",
            "up1",
        ]

        self.start_layer = start_layer
        self.end_layer = end_layer

        self.reduced_layers = self.ordered_layers[self.ordered_layers.index(self.start_layer):(
                    self.ordered_layers.index(self.end_layer) + 1)]

        for k, v in self.layers.items():
            setattr(self, k, v)

    def forwardOnce(self, inputs, pass_no):

        # Use MCDO if Multiple Passes
        mcdo = (self.mcdo_passes > 1)

        if "down1" in self.reduced_layers:
            down1, indices_1, unpool_shape1 = self.layers["down1"](inputs)

        if "down2" in self.reduced_layers:
            down2, indices_2, unpool_shape2 = self.layers["down2"](down1)

        if "down3" in self.reduced_layers:
            if self.fixed_mcdo:
                down3, indices_3, unpool_shape3 = self.layers["down3"](down2)
                if self.training or mcdo:
                    down3 = self.dropout_masks[pass_no]["down3"] * down3
            else:
                down3, indices_3, unpool_shape3 = self.layers["down3"](down2, MCDO=mcdo)
            

        if "down4" in self.reduced_layers:
            if self.fixed_mcdo:
                down4, indices_4, unpool_shape4 = self.layers["down4"](down3)
                if self.training or mcdo:
                    down4 = self.dropout_masks[pass_no]["down4"] * (down4)
            else:
                down4, indices_4, unpool_shape4 = self.layers["down4"](down3, MCDO=mcdo)

        if "down5" in self.reduced_layers:
            if self.fixed_mcdo:
                down5, indices_5, unpool_shape5 = self.layers["down5"](down4)
                if self.training or mcdo:
                    down5 = self.dropout_masks[pass_no]["down5"] * (down5)
            else:
                down5, indices_5, unpool_shape5 = self.layers["down5"](down4, MCDO=mcdo)

        if "up5" in self.reduced_layers:
            if self.fixed_mcdo:
                up5 = self.layers["up5"](down5, indices_5, unpool_shape5)
                if self.training or mcdo:
                    up5 = self.dropout_masks[pass_no]["up5"] * (up5)
            else:
                up5 = self.layers["up5"](down5, indices_5, unpool_shape5, MCDO=mcdo)

        if "up4" in self.reduced_layers:
            if self.fixed_mcdo:
                up4 = self.layers["up4"](up5, indices_4, unpool_shape4)
                if self.training or mcdo:
                    up4 = self.dropout_masks[pass_no]["up4"] * (up4)
            else:
                up4 = self.layers["up4"](up5, indices_4, unpool_shape4, MCDO=mcdo)
            

        if "up3" in self.reduced_layers:
            if self.fixed_mcdo:
                up3 = self.layers["up3"](up4, indices_3, unpool_shape3)
                if self.training or mcdo:
                    up3 = self.dropout_masks[pass_no]["up3"] * (up3)
            else:
                up3 = self.layers["up3"](up4, indices_3, unpool_shape3, MCDO=mcdo)

        if "up2" in self.reduced_layers:
            up2 = self.layers["up2"](up3, indices_2, unpool_shape2)
            # up2 = self.dropouts["up2"](up2)

        if "up1" in self.reduced_layers:
            up1 = self.layers["up1"](up2, indices_1, unpool_shape1)
            # up1 = self.dropouts["up1"](up1)

        # print("inputs",inputs.shape)
        # print("down1",down1.shape)
        # print("down2",down2.shape)
        # print("down3",down3.shape)
        # print("down4",down4.shape)
        # print("down5",down5.shape)
        # print("up1",up1.shape)
        # print("up2",up2.shape)
        # print("up3",up3.shape)
        # print("up4",up4.shape)
        # print("up5",up5.shape)
        # exit()

        return up1

    def configureDropout(self):

        # Determine Type of Dropout
        if self.training:
            for k in self.dropouts.keys():
                self.dropouts[k].train(mode=True)
        else:
            if self.mcdo_passes > 1:
                for k in self.dropouts.keys():
                    self.dropouts[k].train(mode=True)
            else:
                for k in self.dropouts.keys():
                    self.dropouts[k].eval()

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

    def forward(self, inputs, recalType="None"):
        # First pass has backpropagation; others do not
        for i in range(self.mcdo_passes):
            if i == 0:
                x_bp = self.forwardOnce(inputs, i)
                x = x_bp.unsqueeze(-1)
            else:
                with torch.no_grad():
                    x = torch.cat((x, self.forwardOnce(inputs, i).unsqueeze(-1)), -1)

        if self.temperatureScaling:
            x_bp = x_bp / self.temperature
            x = x / self.temperature
        """
        points = ((0,50,50), (0,100,100))
        # plot the distribution of passes
        for n, i, j in points:
            fig, axes = plt.subplots(3, self.n_classes // 3 + 1)
            for c in range(self.n_classes):
                xx = x[n, c, i, j, :]
                axes[(c + 1) // (self.n_classes // 3 + 1), (c + 1) % (self.n_classes // 3 + 1)].hist(xx)
                axes[(c + 1) // (self.n_classes // 3 + 1), (c + 1) % (self.n_classes // 3 + 1)].set_title("Class: {}".format(c))
                del xx

            plt.tight_layout()
            plt.savefig("./{}_{}x{}.png".format(n, i, j))
            plt.close(fig)
        """
        # Uncalibrated Softmax Mean and Variance
        mean = torch.nn.Softmax(1)(x).mean(-1)
        variance = torch.nn.Softmax(1)(x).std(-1)
        if self.recalibrator != "None":
            if recalType == "beforeMCDO":
                for c in range(self.n_classes):
                    x[:, c, :, :, :] = self.calibrationPerClass[c].predict(x[:, c, :, :, :].reshape(-1)).reshape(x[:, c, :, :, :].shape)
                mean = torch.nn.Softmax(1)(x).mean(-1)
                variance = torch.nn.Softmax(1)(x).std(-1)
            elif recalType == "afterMCDO":
                for c in range(self.n_classes):
                    mean[:, c, :, :] = self.calibrationPerClass[c].predict(mean[:, c, :, :].reshape(-1)).reshape(mean[:, c, :, :].shape)

        return x_bp, mean, variance

    def applyCalibration(self, output):

        for c in range(self.n_classes):
            output[:, c, :, :] = self.calibrationPerClass[c].predict(output[:, c, :, :].reshape(-1)).reshape(output[:, c, :, :].shape)

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

        # TODO fix plotting with invalid probabilities and graph wrapping
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

        """
        # Recalibration Curve
        x = np.arange(0, 1, 0.001)
        x = torch.from_numpy(x).float()

        # TODO figure out why we are calibrating the already recalibrated class scores?

        y = self.applyCalibration(x)

        x = x.cpu().numpy()
        y = y.cpu().numpy()

        # y = calibration[m].predict(x)
        # y = calibration[m].predict(x[:,np.newaxis])
        axes[2].plot(x, y)
        axes[2].set_title("Recalibration Curve")
        axes[2].set_xlabel("softmax probability")
        axes[2].set_ylabel("calibrated confidence")
        """

        # calculating expected calibration error
        ECE = sum([abs(i - j) for i,j, in zip(x,y)])/len(x)
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
            axes[(c + 1) // (self.n_classes // 3 + 1), (c + 1) % (self.n_classes // 3 + 1)].set_title("Class: {}".format(c))

        path = "{}/{}/{}".format(logdir, 'calibration', model)
        if not os.path.exists(path):
            os.makedirs(path)
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
            axes[(c + 1) // (self.n_classes // 3 + 1), (c + 1) % (self.n_classes // 3 + 1)].set_title("Class: {}".format(c))

        path = "{}/{}/{}".format(logdir, 'calibration', model)
        if not os.path.exists(path):
            os.makedirs(path)

        plt.tight_layout()
        plt.savefig("{}/calibratedPerClass{}.png".format(path, iteration))
        plt.close(fig)

        torch.cuda.empty_cache()
