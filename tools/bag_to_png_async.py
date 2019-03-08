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

import matplotlib.pyplot as plt

def main():
    """Extract a folder of images from a rosbag.
    """

    output_root = "airsim_data_async"

    topic_folders = [
                     # DroneHigh Left
                     ("/airsim/DroneHigh/vision/left/pose", "pose"),
                     ("/airsim/DroneHigh/vision/left/scene", "scene"),
                     ("/airsim/DroneHigh/vision/left/segmentation", "segmentation"),
                     ("/airsim/DroneHigh/vision/left/DepthPerspective", "depth"),
                     ("/airsim/DroneHigh/vision/left/DepthPerspective_Encoded", "depth_encoded"),

                     # DroneHigh Right
                     ("/airsim/DroneHigh/vision/right/pose", "pose"),
                     ("/airsim/DroneHigh/vision/right/scene", "scene"),
                     ("/airsim/DroneHigh/vision/right/segmentation", "segmentation"),
                     ("/airsim/DroneHigh/vision/right/DepthPerspective", "depth"),
                     ("/airsim/DroneHigh/vision/right/DepthPerspective_Encoded", "depth_encoded"),
                     
                     # DroneLow Back
                     ("/airsim/DroneLow/vision/back/pose", "pose"),
                     ("/airsim/DroneLow/vision/back/scene", "scene"),
                     ("/airsim/DroneLow/vision/back/segmentation", "segmentation"),
                     ("/airsim/DroneLow/vision/back/DepthPerspective", "depth"),
                     ("/airsim/DroneLow/vision/back/DepthPerspective_Encoded", "depth_encoded"),

                     # DroneLow Frone
                     ("/airsim/DroneLow/vision/front/pose", "pose"),
                     ("/airsim/DroneLow/vision/front/scene", "scene"),
                     ("/airsim/DroneLow/vision/front/segmentation", "segmentation"),
                     ("/airsim/DroneLow/vision/front/DepthPerspective", "depth"),
                     ("/airsim/DroneLow/vision/front/DepthPerspective_Encoded", "depth_encoded"),

                     # DroneOverhead Overhead
                     ("/airsim/DroneOverhead/vision/overhead/pose", "pose"),
                     ("/airsim/DroneOverhead/vision/overhead/scene", "scene"),
                     ("/airsim/DroneOverhead/vision/overhead/segmentation", "segmentation"),
                     ("/airsim/DroneOverhead/vision/overhead/DepthPerspective", "depth"),
                     ("/airsim/DroneOverhead/vision/overhead/DepthPerspective_Encoded", "depth_encoded"),
                     ]

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


    for root, dirnames, filenames in os.walk('urban_async'):
        for filename in fnmatch.filter(filenames, '*.bag'):
            file_path = os.path.join(root, filename)
            file_no_ext = filename.split(".")[0]

            print(filename)
            trajectory = root.split("/")[1]
            condition = file_no_ext


            for folder, topic in topic_folders:

                print('list num',len(folder.split('/')))
                
                print(folder.split("/"))

                if not "pose" in folder.split('/'): # segmentation, depth, scene

                    camera_id = folder.split('/')[4]
                    output_dir = "{}/{}/{}/{}/{}".format(output_root,topic,condition,trajectory,camera_id)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    bag = rosbag.Bag(file_path, "r")
                    bridge = CvBridge()

                    count = 0
                    print('topic',topic)
                    for topic, msg, t in bag.read_messages(topics=[folder]):

                        print('topic',topic)
                        print('msg',type(msg))
                        print('t',t)
                        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                        # if "Encoded" in topic:
                        #     print(topic)
                        #     print(cv_img.shape)
                        #     plt.figure()
                        #     plt.subplot(2,1,1)
                        #     plt.imshow(cv_img[:,:,:3])
                        #     plt.subplot(2,1,2)
                        #     plt.imshow((256**3)*cv_img[:,:,0]+
                        #                (256**2)*cv_img[:,:,1]+
                        #                (256**1)*cv_img[:,:,2]+
                        #                (256**0)*cv_img[:,:,3])
                        #     plt.show()

                        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)

                        print("{} {} {} ({})".format(file_path, topic, output_dir, count))

                        count += 1

                else: # pose


                    camera_id = folder.split('/')[4]
                    output_dir = "{}/{}/{}/{}/{}".format(output_root,topic,condition,trajectory,camera_id)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    bag = rosbag.Bag(file_path, "r")
                    bridge = CvBridge()

                    count = 0
                    print('topic',topic)
                    for topic, msg, t in bag.read_messages(topics=[folder]):

                        print('topic',topic)
                        print('msg',(msg.pose))
                        print('t',t)

                        pose_string = str(msg.pose.position.x)+' '+str(msg.pose.position.y)+' '+str(msg.pose.position.z)
                        orient_string = str(msg.pose.orientation.x)+' '+str(msg.pose.orientation.y)+' '+str(msg.pose.orientation.z)+' '+str(msg.pose.orientation.w)

                        with open(os.path.join(output_dir, "pose%06i.txt" % count), 'w') as file:
                            file.write(pose_string+' '+orient_string)

                        #cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                        #cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)

                        print("{} {} {} ({})".format(file_path, topic, output_dir, count))

                        count += 1



            bag.close()



    return

if __name__ == '__main__':
    main()