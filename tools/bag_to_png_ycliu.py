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

    output_root = "airsim_data"

    topic_folders = [("/airsim/Drone1/pose", "pose"),
                     ("/airsim/Drone1/vision/back_upper/Scene", "scene"),
                     ("/airsim/Drone1/vision/back_upper/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/back_upper/DepthPerspective", "depth"),
                     


                     ("/airsim/Drone1/vision/front_upper/Scene", "scene"),
                     ("/airsim/Drone1/vision/front_upper/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/front_upper/DepthPerspective", "depth"),

                     ("/airsim/Drone1/vision/left_upper/Scene", "scene"),
                     ("/airsim/Drone1/vision/left_upper/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/left_upper/DepthPerspective", "depth"),

                     ("/airsim/Drone1/vision/right_upper/Scene", "scene"),
                     ("/airsim/Drone1/vision/right_upper/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/right_upper/DepthPerspective", "depth"),
                    
                     ("/airsim/Drone1/vision/back_lower/Scene", "scene"),
                     ("/airsim/Drone1/vision/back_lower/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/back_lower/DepthPerspective", "depth"),
                     
                     ("/airsim/Drone1/vision/front_lower/Scene", "scene"),
                     ("/airsim/Drone1/vision/front_lower/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/front_lower/DepthPerspective", "depth"),

                     ("/airsim/Drone1/vision/left_lower/Scene", "scene"),
                     ("/airsim/Drone1/vision/left_lower/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/left_lower/DepthPerspective", "depth"),

                     ("/airsim/Drone1/vision/right_lower/Scene", "scene"),
                     ("/airsim/Drone1/vision/right_lower/Segmentation", "segmentation"),
                     ("/airsim/Drone1/vision/right_lower/DepthPerspective", "depth"),

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


    for root, dirnames, filenames in os.walk('urban_dense'):
        for filename in fnmatch.filter(filenames, '*.bag'):
            file_path = os.path.join(root, filename)
            file_no_ext = filename.split(".")[0]

            print(filename)
            trajectory = root.split("/")[1]
            condition = file_no_ext


            for folder, topic in topic_folders:

                print('list num',len(folder.split('/')))
                

                if len(folder.split('/'))==6: # segmentation, depth, scene

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

                        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)

                        print("{} {} {} ({})".format(file_path, topic, output_dir, count))

                        count += 1

                else: # pose

                    output_dir = "{}/{}/{}/{}".format(output_root,topic,condition,trajectory)
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