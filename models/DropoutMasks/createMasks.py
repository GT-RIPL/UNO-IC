import torch.nn as nn
from torch.autograd import Variable
import pickle
import torch
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class MaskCreator():
    def __init__(self,dropoutP,device):
        self.dropoutP = dropoutP
        self.device = device


    def sampleDropoutMasks(self):
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

        masks = {
            "down3": Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(256,64,64))).to(self.device),
            "down4": Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(512,32,32))).to(self.device),
            "down5": Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(512,16,16))).to(self.device),
            "up5":   Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(512,32,32))).to(self.device),
            "up4":   Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(256,64,64))).to(self.device),
            "up3":   Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(128,128,128))).to(self.device),
        } 

        # masks = {p:
        # {
        #     "down3": Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(1,256,64,64))).to(self.device).repeat(self.batch_size,1,1,1),
        #     "down4": Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(1,512,32,32))).to(self.device).repeat(self.batch_size,1,1,1),
        #     "down5": Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(1,512,16,16))).to(self.device).repeat(self.batch_size,1,1,1),
        #     "up5":   Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(1,512,32,32))).to(self.device).repeat(self.batch_size,1,1,1),
        #     "up4":   Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(1,256,64,64))).to(self.device).repeat(self.batch_size,1,1,1),
        #     "up3":   Variable((1./(1-self.dropoutP))*torch.bernoulli((1-self.dropoutP)*torch.ones(1,128,128,128))).to(self.device).repeat(self.batch_size,1,1,1),
        # } for p in range(self.mcdo_passes)}

        return masks



if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subsample = 5000

    mc = MaskCreator(0.5,device)

    for i in range(512):

        # path = "models/DropoutMasks/{}/{}".format("Random",i)
        path = "{}/{}".format("Random",i)
        if not os.path.exists(path):
            os.makedirs(path)


        dict_file = "masks.pkl"
        if os.path.isfile("{}/{}".format(path,dict_file)):
            ###############
            # MASK EXISTS #
            ###############            
            print("{}/{}: Exists".format(path,dict_file))
            with open("{}/{}".format(path,dict_file),"rb") as f:
                masks = pickle.load(f)
        else:
            #######################
            # MASK DOES NOT EXIST #
            #######################            
            print("{}/{}: Does Not Exist".format(path,dict_file))
            masks = mc.sampleDropoutMasks()
            with open("{}/{}".format(path,dict_file),"wb") as f:
                pickle.dump(masks,f)

            fig = plt.figure()
            for i,k in enumerate(masks.keys()):
                plt.subplot(2,3,i+1)
                idxSum =  masks[k].sum(0).cpu().numpy()
                plt.imshow(idxSum)
                plt.title(k)
            plt.savefig("{}/MaskSummary.png".format(path))
            plt.close(fig)



        # idx = masks["down3"].nonzero().cpu().numpy()
        # fig = plt.figure()
        # ax = fig.add_subplot(111,projection="3d")
        # ax.scatter( idx[::subsample,0], idx[::subsample,1], idx[::subsample,2] )
        # plt.show()
