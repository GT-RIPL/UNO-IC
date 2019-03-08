import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

root = {}
root['rgb'] = 'scene'
root['depth'] = 'depth'
root['seg'] = 'segmentation'

noise_amount = 10
corrupt_channels = ['rgb','depth']

for corrupt_channel in corrupt_channels:

    corruption_postpend = "{}_additiveNoiseMag{}".format(corrupt_channel,noise_amount)

    for file in glob.glob("{}/**/*.png".format(root[corrupt_channel]),recursive=True):

        print(file)

        mode = file.split("/")[0]
        world = file.split("/")[1]
        condition = file.split("/")[2]
        trajectory = file.split("/")[3]
        frame = file.split("/")[4]
        
        if condition != "fog_000":
            continue


        orig = cv2.imread(file)
        corr = orig.copy()


        m = (noise_amount,noise_amount,noise_amount) 
        s = (noise_amount,noise_amount,noise_amount)
        corr = cv2.randn(corr,m,s)

        img = np.clip(orig + corr,0,255)

        # plt.figure()
        # plt.imshow(img+im1)
        # plt.show()
        
        original_path = {}
        corrupt_path = {}
        for k in root.keys():
            original_path[k] = "{}/{}/{}/{}".format(root[k],world,condition,trajectory)
            corrupt_path[k] = "{}/{}/{}__{}/{}".format(root[k],world,condition,corruption_postpend,trajectory)
        
            if not os.path.exists(corrupt_path[k]):
                os.makedirs(corrupt_path[k])

        print("Corrupting {} -> {}".format(file,corrupt_path[corrupt_channel]))


        cv2.imwrite(corrupt_path[corrupt_channel]+"/"+frame,img)

        for k in corrupt_path.keys():



            if k != corrupt_channel:
                os.system('cp {} {}'.format(original_path[k]+"/"+frame,corrupt_path[k]+"/"+frame))

