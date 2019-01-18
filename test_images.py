import os
from ptsemseg.loader import get_loader, get_data_path
import cv2
import numpy as np
import matplotlib.pyplot as plt

loader = get_loader("airsim")



root = '../../ros/data/airsim/scene/urban'
conditions = ['fog_000','fog_005','fog_010','fog_020','fog_025','fog_050','fog_100']
trains = loader.split_subdirs['train']
tests = loader.split_subdirs['val']
models = ['fcn8s_airsim_fog_all_01-15-2019.pkl']
test = tests[0]

for model in models:
    for test in tests+trains:
        for condition in conditions:
            for i in range(0,10000,100):
                number = "{:06d}".format(i)

                out_dir = "results/urban/{}/{}".format(model,test)
                out_path_dt = out_dir+"/frame{}_condition{}_dt.png".format(number,condition)
                out_path_gt = out_dir+"/frame{}_condition{}_gt.png".format(number,condition)
                out_path_orig = out_dir+"/frame{}_condition{}_orig.png".format(number,condition)
                out_path_raw = out_dir+"/frame{}_condition{}_raw.png".format(number,condition)
                img_path_scene = "../../ros/data/airsim/scene/"+"urban/{}/{}/frame{}.png".format(condition,test,number)
                img_path_seg = "../../ros/data/airsim/segmentation/"+"urban/{}/{}/frame{}.png".format(condition,test,number)

                if not os.path.isfile(img_path_scene):
                    continue

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                print(loader.name2color)

                os.system("CUDA_VISIBLE_DEVICES=1 \
                           python test.py \
                           --model_path models/{} \
                           --dataset airsim \
                           --out_path {} \
                           --img_path {}".format(model,out_path_dt,img_path_scene))


                temp = cv2.imread(img_path_seg)[:,:,::-1]

                for name,colors in loader.name2color.items():
                    for color in colors:

                        print(name,color,"->",colors[0])

                        mask = cv2.inRange(temp,np.array(color),np.array(color))
                        temp[mask!=0] = colors[0]


                # plt.figure()
                # plt.subplot(211)
                # plt.imshow(temp)
                # plt.show()

                cv2.imwrite(out_path_gt,temp[:,:,::-1])                
                os.system("cp {} {}".format(img_path_scene,out_path_raw))
                os.system("cp {} {}".format(img_path_seg,out_path_orig))

                fig = plt.figure()
                plt.subplot(141)
                plt.imshow(cv2.imread(out_path_raw))
                plt.xticks([])
                plt.yticks([])

                plt.xlabel("Input ({})".format(condition))
                plt.subplot(142)
                plt.imshow(cv2.imread(out_path_orig))
                plt.xlabel("GT")
                plt.xticks([])
                plt.yticks([])

                plt.subplot(143)
                plt.imshow(cv2.imread(out_path_gt))
                plt.xlabel("GT-Reduced")
                plt.xticks([])
                plt.yticks([])

                plt.subplot(144)
                plt.imshow(cv2.imread(out_path_dt))
                plt.xlabel("DT")
                plt.xticks([])
                plt.yticks([])

                fig.savefig("{}/{}_{}.png".format(out_dir,i,condition), bbox_inches="tight")

                # plt.show()


                # out_path_gt
                # out_path_dt
                # out_path_orig
                # out_path_raw

                # cv2.imread(out_path_orig)
