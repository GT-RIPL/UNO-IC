import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

root = {}
root['rgb'] = 'scene'
root['depth'] = 'depth'
root['seg'] = 'segmentation'

corrupt_channels = ['rgb','depth']

for corrupt_channel in corrupt_channels:

    corruption_postpend = "{}_occlusion".format(corrupt_channel)

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
        corr = np.ones(orig.shape,dtype=np.uint8)

        x = int(orig.shape[0]*np.random.rand())
        y = int(orig.shape[1]*np.random.rand())
        r = int((min(orig.shape[:2])/4)*np.random.rand()+(min(orig.shape[:2])/4))

        cv2.circle(corr,(x,y),r,0,-1)

        print(x,y,r)


        img = np.clip(orig*corr,0,255)

        # plt.figure()
        # plt.imshow(img)
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

