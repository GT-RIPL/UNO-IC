import matplotlib
matplotlib.use('Agg')

import os
import torch
import numpy as np
import scipy.misc as m
import glob
import cv2
import time
import matplotlib.pyplot as plt
import copy
from random import shuffle
import random
from torch.utils import data
import yaml
from tqdm import tqdm
import pickle
from ptsemseg.degredations import *
random.seed(42)

import ptsemseg.augmentations.augmentations as aug

"""
synthia-rand
Class		R	G	B	ID
void		0	0	0	0
sky		70	130	180	1
Building	70	70	70	2
Road		128	64	128	3
Sidewalk	244	35	232	4
Fence		64	64	128	5
Vegetation	107	142	35	6
Pole		153	153	153	7
Car		0	0	142	8
Traffic sign	220	220	0	9
Pedestrian	220	20	60	10
Bicycle		119	11	32	11
Motorcycle	0	0	230	12
Parking-slot	250	170	160	13
Road-work	128	64	64	14
Traffic light	250	170	30	15
Terrain		152	251	152	16
Rider		255	0	0	17
Truck		0	0	70	18
Bus		0	60	100	19
Train		0	80	100	20
Wall		102	102	156	21
Lanemarking	102	102	156	22

synthia-seq
Class		R	G	B	ID
Void		0 	0 	0	0
Sky             128 	128 	128	1
Building        128 	0 	0	2
Road            128 	64 	128	3
Sidewalk        0 	0 	192	4
Fence           64 	64 	128	5
Vegetation      128 	128 	0	6
Pole            192 	192 	128	7
Car             64 	0 	128	8
Traffic Sign    192 	128 	128	9
Pedestrian      64 	64 	0	10
Bicycle         0 	128 	192	11
Lanemarking	0	172 	0	12
Reserved	- 	- 	-	13
Reserved	- 	- 	-	14
Traffic Light	0 	128 	128	15

"""

class synthiaLoader(data.Dataset):

    class_names = np.array([
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "light",
        "sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motocycle",
        "bicycle",
    ])
    
    image_modes = ['RGB', 'Depth', 'GT/COLOR', 'GT/LABELS']
    sides = ['Stereo_Left','Stereo_Right']
    cam_pos = ['Omni_B','Omni_F','Omni_L','Omni_R']

    split_subdirs = {}
    ignore_index = 0
    mean_rgbd = {
        # "airsim": [41.454376, 46.093113, 42.958637, 4.464941, 5.1877136, 167.58365] # joint 000 050 fog
        # "synthia-rand": [36.8617, 42.363132, 38.955276, 4.8034225, 4.503343, 167.34187] # 000 fog
        "synthia-seq": [55.09944,  62.203827, 71.23802 , 130.35643,1.8075644,15.805721] # synthia-seq
    }  # pascal mean for PSPNet and ICNet pre-trained model

    std_rgbd = {
        # "airsim": [37.94737, 37.26296, 36.74846, 22.874805, 28.264046, 39.39389] # joint 000 050 fog
        # "synthia": [47.416595,48.246918,47.81453, 23.966692, 25.054394, 41.507214] # 000 fog
        "synthia-seq": [49.56111,  51.497387, 55.363934 , 46.930763, 10.479317, 34.19771] # synthia-seq   
    }
    
    # mean_rgb = [36.8617, 42.363132, 38.955276]
    # mean_d =  [4.8034225, 4.503343, 167.34187]

    # std_rgb =  [47.416595,48.246918,47.81453] 
    # std_d =  [23.966692, 25.054394, 41.507214] 
    def __init__(
        self,
        root,
        split="train",
        subsplits=None,
        is_transform=False,
        img_size=(512, 512),
        scale_quantity=1.0,      
        img_norm=True,
        version='synthia-seq',   
        augmentations=None,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """

        self.root = root
        self.split = split
        self.subsplits = subsplits
        self.is_transform = is_transform
        self.img_norm = img_norm
        self.n_classes = len(self.class_names)
        self.img_size = (img_size if isinstance(img_size, tuple) else (img_size, img_size))
        
        # split: train/val image_modes
        self.imgs = {image_mode:[] for image_mode in self.image_modes}
        self.dgrd = {image_mode:[] for image_mode in self.image_modes}
        self.mean = np.array(self.mean_rgbd[version])
        self.std = np.array(self.std_rgbd[version])



        # load RGB/Depth
        for subsplit in self.subsplits:
            if len(subsplit.split("__")) == 2:
                condition = subsplit.split("__")[0]
                degradation = subsplit.split("__")[1]
            else:
                condition = subsplit
                degradation = None

            for comb_modal in self.image_modes:
                for comb_cam in self.cam_pos:
                    for side in self.sides:
                        files = glob.glob(os.path.join(root,condition,comb_modal,side,comb_cam,'*.png'),recursive=True)
                        random.seed(0)
                        shuffle(files)
                        
                        # print(os.path.join(root,condition,comb_modal,side,comb_cam,'*.png'))
                        # print(len(files))
                        n = len(files)
                        n_train = int(0.6 * n)
                        n_recal = int(0.1 * n)
                        n_valid = int(0.1 * n)
                        n_test = int(0.2 * n)
                        
                        if self.split == 'train':
                            files = files[:n_train]
                        if self.split == 'recal':
                            files = files[n_train:n_train+n_recal]
                        if self.split == 'valid':
                            files = files[n_train+n_recal:n_train+n_recal+n_valid]
                        if self.split == 'test':
                            files = files[n_train+n_recal+n_valid:]
                        
                        
                        for file_path in files:
                            self.imgs[comb_modal].append(file_path)
                            self.dgrd[comb_modal].append(degradation)
        
        
        
        if not self.imgs[self.image_modes[0]]:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))

        print("{} {}: Found {} Images".format(self.split,self.subsplits,len(self.imgs[self.image_modes[0]])))
        if scale_quantity != 1.0:
            for image_mode in self.image_modes:
                self.imgs[image_mode] = self.imgs[image_mode][::int(1/scale_quantity)]
            print("{} {}: Reduced by {} to {} Images".format(self.split,self.subsplits,scale_quantity,len(self.imgs[self.image_modes[0]])))


        # self.dataset_statistics()
        # exit()

    def dataset_statistics(self):

        print("="*20)
        print("Running Dataset Statistics")
        print("="*20)

        print("Splits:    {}".format(self.split))
        print("Positions: {}".format(", ".join(list(self.imgs.keys()))))

        savefile = "{}_dataset_statistics.p".format(self.split)
        savefile2 = "{}_pixels.p".format(self.split)


        rgb_mean = []
        rgb_std = []
        d_mean = []
        d_std = []
        
        from collections import defaultdict
        
        classes = defaultdict(int)
        
        for index in tqdm(range(int(1.0*len(self.imgs[self.image_modes[0]])))):
            input_list, lbl_list = self.__getitem__(index)
            img_list, aux_list = input_list['rgb'], input_list['d']
            # shape (batch_size, 3, height, width)
            numpy_image = torch.stack(img_list, 0).numpy()
            numpy_depth = torch.stack(aux_list, 0).numpy()
            
            # shape (3,)
            batch_mean = np.mean(numpy_image, axis=(0,2,3))
            batch_std = np.std(numpy_image, axis=(0,2,3), ddof=1)
            
            rgb_mean.append(batch_mean)
            rgb_std.append(batch_std)
            
            
            # shape (3,)
            batch_mean = np.mean(numpy_depth, axis=(0,2,3))
            batch_std = np.std(numpy_depth, axis=(0,2,3), ddof=1)
            
            d_mean.append(batch_mean)
            d_std.append(batch_std)
            
            
            numpy_lbl = torch.stack(lbl_list, 0).reshape(-1).numpy()
            for l in numpy_lbl:
            
                classes[l] += 1

            

        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        rgb_mean = np.array(rgb_mean).mean(axis=0)
        rgb_std = np.array(rgb_std).mean(axis=0)
        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        d_mean = np.array(d_mean).mean(axis=0)
        d_std = np.array(d_std).mean(axis=0)
        
        print("rgb: mean - {}, std - {}".format(rgb_mean, rgb_std))
        print("d: mean - {}, std - {}".format(d_mean, d_std))
        print(classes)
        # if os.path.isfile(savefile):
        if False:
            pixel_stats = pickle.load( open(savefile,"rb"))
            pixel_dump = pickle.load( open(savefile2,"rb"))

        else:        
            pixel_stats = {p:{n:[] for n in self.name2id} for p in self.cam_pos}
            
            for index in tqdm(range(int(1.0*len(self.imgs[self.image_modes[0]])))):
                input_list, lbl_list = self.__getitem__(index)
                img_list, aux_list = input_list['rgb'], input_list['d']
                
                for i,p in enumerate(self.cam_pos):
                    for n in self.name2id:
                        pixel_stats[p][n].append( (1.*torch.sum(lbl_list[i]==self.name2id[n]).tolist()/(list(lbl_list[i].size())[0]*list(lbl_list[i].size())[1])) ) 
                
            
            pickle.dump(pixel_stats, open(savefile,"wb"))
        
        pixel_stats_summary = {p:{n:{"mean":0} for n in self.name2id} for p in self.cam_pos}
        for i,p in enumerate(self.cam_pos):
            for n in self.name2id:
                pixel_stats_summary[p][n]["mean"] = np.mean(pixel_stats[p][n])
                pixel_stats_summary[p][n]["var"] = np.std(pixel_stats[p][n])

        # print(pixel_stats_summary)



        # for cam_pos in self.cam_pos:
        #     for image_mode in self.image_modes:
        #         for index in range(len(self.imgs[cam_pos][image_mode])):
        #             print(index)



    def tuple_to_folder_name(self, path_tuple):
        start = path_tuple[1]
        end = path_tuple[2]
        path=str(start[0])+'_'+str(-start[1])+'__'+str(end[0])+'_'+str(-end[1])

        return path


    def __len__(self):
        """__len__"""
        return len(self.imgs[self.image_modes[0]])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        input_list = {'rgb':[], 
                      'd': [],
                      'rgb_display': [],
                      'd_display': []}
        lbl_list = []
        start_ts = time.time()

        img_path = self.imgs['RGB'][index]
        lbl_path = self.imgs['GT/LABELS'][index]
        lbl_color_path = self.imgs['GT/COLOR'][index]            
        
        img = np.array(cv2.imread(img_path),dtype=np.uint8)[:,:,:3]
        lbl = np.array(cv2.imread(lbl_path,cv2.IMREAD_UNCHANGED))[:,:,2]
        lbl_color = np.array(cv2.imread(lbl_color_path),dtype=np.uint8)[:,:,:3]
        

        depth_path = self.imgs['Depth'][index]
        # depth = np.array(cv2.imread(depth_path),dtype=np.uint8)
        
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth = np.array(cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        # cv2.imwrite('messigray.png',depth)
        degradation = self.dgrd['RGB'][index]
        if not degradation is None:
            img, depth = self.degradation(degradation, img, depth)

        if self.is_transform:
            img, lbl, depth, img_display, depth_display = self.transform(img, lbl, depth)
        input_list['rgb'].append(img)
        input_list['d'].append(depth)
        input_list['rgb_display'].append(img_display)
        input_list['d_display'].append(depth_display)
        
        lbl_list.append(lbl)
        

        return input_list, lbl_list


    def degradation(self, degradation, img, depth):

        degradation = yaml.load((degradation))

        if degradation['type'] in key2deg.keys():
            if "rgb" in degradation['channel']:
                img = key2deg[degradation['type']](img, int(degradation['value']))
            if "d" in degradation['channel']:
                depth = key2deg[degradation['type']](depth, int(degradation['value']))
        else:
            print("Corruption Type Not Implemented")
            
        return img, depth


    def transform(self, img, lbl, aux):
        """transform

        :param img:
        :param lbl:
        """
        # img = Image.fromarray(img, 'RGB')
        # aux = Image.fromarray(aux, 'RGB')
        # lbl = Image.fromarray(lbl)
        
        # if self.split == 'train' or self.split == 'recal':
            # sample = aug.transform_tr({'image':img, 'aux':aux, 'label':lbl}, self.mean_rgb, self.mean_d, self.std_rgb, self.std_d)
        # if self.split == 'val':
            # sample = aug.transform_val({'image':img, 'aux':aux, 'label':lbl}, self.mean_rgb, self.mean_d, self.std_rgb, self.std_d)
        # if self.split == 'test':
            # sample = aug.transform_ts({'image':img, 'aux':aux, 'label':lbl}, self.mean_rgb, self.mean_d, self.std_rgb, self.std_d)

        # return sample['image'], sample['label'], sample['aux'], sample['img_display'], sample['aux_display']

        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        aux = cv2.resize(aux, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        # img = img[:, :, ::-1]  # RGB -> BGR

        img = img.astype(np.float64)
        aux = aux.astype(np.float64)
        
        img_display = img.copy()
        aux_display = aux.copy()

        if self.img_norm:
            img = np.divide((img.astype(float) - self.mean[:3]),self.std[:3])
            aux = np.divide((aux.astype(float) - self.mean[3:]),self.std[3:])

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img_display = img_display.transpose(2, 0, 1)

        if not any(['depth_encoded'==mode for mode in self.image_modes]):
            aux = aux.transpose(2, 0, 1)
            aux_display = aux_display.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST) #, "nearest", mode="F")
        lbl = lbl.astype(int)

        # if not np.all(classes == np.unique(lbl)):
        #     print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        img = torch.from_numpy(img).float()
        aux = torch.from_numpy(aux).float()
        img_display = torch.from_numpy(img_display).float()
        aux_display = torch.from_numpy(aux_display).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl, aux, img_display, aux_display


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt


    local_path = "/home/datasets/synthia-seq/"

    dst = airsimLoader(local_path, is_transform=True,split='val') #, augmentations=augmentations)
    bs = 4

    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels  = data
        #print(len(imgs))
        #print(len(labels))
        #print(len(aux))
        # import pdb;pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(labels.numpy()[j])
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()