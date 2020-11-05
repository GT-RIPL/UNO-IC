import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

root = {}
root['rgb'] = 'scene'
root['depth'] = 'depth'
root['seg'] = 'segmentation'

noise_amount = 20
corrupt_channels = ['rgb','depth']

print("HI")

for corrupt_channel in corrupt_channels:

    corruption_postpend = "{}_blackoutNoiseMag{}".format(corrupt_channel,noise_amount)

    for file in glob.glob("{}/**/*.png".format(root[corrupt_channel]),recursive=True):

        print(file)

        mode = file.split("/")[0]
        # world = file.split("/")[1]
        condition = file.split("/")[1]
        trajectory = file.split("/")[2]
        camera = file.split("/")[3]
        frame = file.split("/")[4]
        
        print(condition)

        if condition != "8camera_fog_100_dense":
            continue


        orig = cv2.imread(file)
        corr = orig.copy()

        corr = np.zeros(orig.shape,dtype=np.uint8)

        m = (noise_amount,noise_amount,noise_amount) 
        s = (noise_amount,noise_amount,noise_amount)
        corr = cv2.randn(corr,m,s)



        # img = np.clip(orig + corr,0,255)
        img = np.clip(corr,0,255)

        # plt.figure()
        # plt.imshow(img+im1)
        # plt.show()
        
        original_path = {}
        corrupt_path = {}
        for k in root.keys():
            # original_path[k] = "{}/{}/{}/{}".format(root[k],world,condition,trajectory)
            original_path[k] = "{}/{}/{}/{}".format(root[k],condition,trajectory,camera)
            corrupt_path[k] = "{}/{}__{}/{}/{}".format(root[k],condition,corruption_postpend,trajectory,camera)
        
            if not os.path.exists(corrupt_path[k]):
                os.makedirs(corrupt_path[k])

        print("Corrupting {} -> {}".format(file,corrupt_path[corrupt_channel]))


        cv2.imwrite(corrupt_path[corrupt_channel]+"/"+frame,img)

        for k in corrupt_path.keys():
            if k != corrupt_channel:
                os.system('cp {} {}'.format(original_path[k]+"/"+frame,corrupt_path[k]+"/"+frame))

