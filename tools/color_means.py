#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import matplotlib
matplotlib.use('TkAgg')

import os
import fnmatch
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import glob

import matplotlib.pyplot as plt

import numpy as np

def main():
    """Extract a folder of images from a rosbag.
    """

    output_root = "airsim_data"


    averages = {}

    i = 0

    for root, dirnames, filenames in os.walk('airsim_data'):
        for filename in fnmatch.filter(filenames, '*.png'):

            i+=1
            if i % 1e3 != 0:
                continue
            else:
                i = 0



            file_path = os.path.join(root, filename)
            file_no_ext = filename.split(".")[0]

            mode, condition, trajectory, camera = root.split("/")[1:]



            if not mode in averages.keys():
                averages[mode] = {}
            if not condition in averages[mode].keys():
                averages[mode][condition] = []

            # if len(averages[mode][condition])>100:
            #     continue


            img = cv2.imread(file_path)

            # plt.figure()
            # plt.imshow(img)
            # plt.show()

            avg = np.divide(np.sum(img,axis=(0,1)),1.*np.prod(list(img.shape[:2])))
            # print("{} {} {} {}: {}".format(mode,condition,trajectory,camera,avg))


            averages[mode][condition].append([avg])

            print("{} {}: {} {}".format(mode,condition,np.mean(np.array(averages[mode][condition]),axis=0),np.std(np.array(averages[mode][condition]),axis=0)))


            # print(filename,mode,condition,trajectory,)

    # average lists
    for mode in averages.keys():
        for condition in averages[mode].keys():
            averages[mode][condition] = (np.mean(np.array(averages[mode][condition]),axis=0),np.std(np.array(averages[mode][condition]),axis=0))

    print(averages)


    return

if __name__ == '__main__':
    main()