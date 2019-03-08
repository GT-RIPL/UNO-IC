#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import fnmatch
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import glob

def main():
    """Extract a folder of images from a rosbag.
    """

    output_root = "airsim"

    topic_folders = [("/airsim/Drone1/vision/Scene", "scene"),
                     ("/airsim/Drone1/vision/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/DepthPerspective", "depth")]

    # for file_path in glob.glob("bag/**/*.bag",recursive=True):

    #     print(file_path)

    #     # file_name = file_path.split("/")[-1].split(".")[0]

    #     # for topic, folder in topic_folders:
    #     #     output_dir = folder+"/"+file_name
    #     #     if not os.path.exists(output_dir):
    #     #         os.makedirs(output_dir)



    #     #     # bag = rosbag.Bag(file_path, "r")
    #     #     # bridge = CvBridge()
    #     #     # count = 0
    #     #     # for topic, msg, t in bag.read_messages(topics=[topic]):
    #     #     #     cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    #     #     #     cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)

    #     #     #     print("{} {} {} ({})".format(file_path, topic, output_dir, count))

    #     #     #     count += 1

    #     #     bag.close()


    for root, dirnames, filenames in os.walk('bag'):
        for filename in fnmatch.filter(filenames, '*.bag'):
            file_path = os.path.join(root, filename)
            file_no_ext = filename.split(".")[0]
            environment = root.split("/")[1]
            trajectory = root.split("/")[2]
            condition = file_no_ext

            # print(file_path)
            # continue


            # if "fog_100" in filename: # or "fog_050" in filename:
            #     continue



            for topic, folder in topic_folders:
                output_dir = "{}/{}/{}/{}/{}".format(output_root,folder,environment,condition,trajectory)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                bag = rosbag.Bag(file_path, "r")
                bridge = CvBridge()
                count = 0
                for topic, msg, t in bag.read_messages(topics=[topic]):
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                    cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)

                    print("{} {} {} ({})".format(file_path, topic, output_dir, count))

                    count += 1

            bag.close()



    return

if __name__ == '__main__':
    main()