import os
import torch
import numpy as np
import scipy.misc as m
import glob

from torch.utils import data

import matplotlib.pyplot as plt

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


class airsimLoader(data.Dataset):
    
    name2color = {"person":  [[153, 108, 6  ]],
                  "road":    [[53,  32,  65 ], [115, 176, 195], [37,  128, 125], [ 49,  89, 160]],
                  "building":[[84,  226, 87 ], [55,  181, 57 ], [22,  61,  247], [230, 136, 198], [190, 225, 64 ]], 
                  "vehicle": [[161, 171, 27 ], [89,  121, 72 ]],
                  "tree":    [[147, 144, 233], [112, 105, 191]],
                  "sidewalk":[[82,  239, 232], [178, 221, 213], [27,  8,   38], [170, 179, 42 ], [ 78,  98, 254], [223, 248, 167]],
                  "grass":   [[81,  13,  36 ]],
                  "sky":     [[209, 247, 202]]  }

    name2id = {"person": 1, "vehicle": 2, "road": 3, "building": 4, "sidewalk": 5, "tree": 6, "grass": 7, "sky": 8}
    id2name = {i:name for name,i in name2id.items()}

    splits = ['train','val']
    image_modes = ['scene','depth','segmentation']

    split_subdirs = {}
    # split_subdirs['train'] = ['0_0__-209_-22', '-130_-255__-148_-130', '-219_68__-302_-24', '-302_-24__-309_-172', '112_-272__115_-21', '-209_-22__-48_-193', '-309_-172__-219_-264', '115_-21__62_76', '-219_-264__-130_-255', '-48_-193__112_-272']
    # split_subdirs['val'] = ['-94_58__-143_39', '0_0__-94_58', '-143_39__-219_68']
        
    split_subdirs['train'] = [ 
                            "0_0__-94_58",
                            "123_-135__216_-26",
                            "-130_-255__-57_-255",
                            "-143_39__-219_68",
                            "176_-355__250_-180",
                            "-302_-24__-309_-172",
                            "-309_-172__-219_-264",
                            "-57_-255__88_-202",
                            "88_-202__20_-355",
                            "-94_58__-143_39",]

    split_subdirs['val'] = [
                            "20_-355__176_-355",
                            "-219_-264__-130_-255",
                            "-219_68__-302_-24",
                            "250_-180__123_-135",]


    ignore_index = 0

    mean_rgbd = {
        "airsim": [103.939, 116.779, 123.68, 120.00],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
        subsplit=None,
        is_transform=False,
        img_size=(512, 512),
        augmentations=None,
        img_norm=True,
        version="airsim",
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
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 9
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array(self.mean_rgbd[version])
        
        self.imgs = {s:{image_mode:[] for image_mode in self.image_modes} for s in self.splits}
        for split in self.splits:
            for subdir in self.split_subdirs[split]:
                for file_path in glob.glob(os.path.join(root,'scene/**',subdir,"**/*.png"),recursive=True):
                    ext = file_path.replace(root+"/scene/",'')
                    env = ext.split("/")[1]

                    if all([os.path.exists(os.path.join(root,image_mode,ext)) for image_mode in self.image_modes]):
                        if subsplit is None or (not subsplit is None and subsplit==env):
                            [self.imgs[split][image_mode].append(os.path.join(root,image_mode,ext)) for image_mode in self.image_modes]


        if not self.imgs[self.split][self.image_modes[0]]:
            raise Exception(
                "No files for split=[%s] found in %s" % (self.split, self.root)
            )

        print("Found %d %s images" % (len(self.imgs[self.split][self.image_modes[0]]), self.split))

    def __len__(self):
        """__len__"""
        return len(self.imgs[self.split][self.image_modes[0]])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """

        img_path, mask_path = self.imgs[self.split]['scene'][index], self.imgs[self.split]['segmentation'][index]
        img, mask = np.array(m.imread(img_path),dtype=np.uint8)[:,:,:3], np.array(m.imread(mask_path),dtype=np.uint8)[:,:,:3]

        depth_path = self.imgs[self.split]['depth'][index]
        depth = np.array(m.imread(depth_path),dtype=np.uint8)[:,:,0]

        # aux = np.dstack((depth,depth))
        aux = depth

        # plt.figure()
        # plt.imshow(depth)
        # plt.show()

        lbl = self.ignore_index*np.ones((img.shape[0],img.shape[1]),dtype=np.uint8)
        for i,name in self.id2name.items():
            for color in self.name2color[name]:
                lbl[(mask==color).all(-1)] = i

        if self.augmentations is not None:
            img, lbl, aux = self.augmentations(img, lbl, aux)

        if self.is_transform:
            img, lbl, aux = self.transform(img, lbl, aux)

        return img, lbl, aux

    def transform(self, img, lbl, aux):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        aux = aux.astype(np.float64)
        img -= self.mean[:3]
        aux -= self.mean[3:]
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
            aux = aux.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        # aux = aux.astype(float)
        # aux = m.imresize(aux, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        # aux = aux.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        aux = torch.from_numpy(aux).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl, aux

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for i,name in self.id2name.items():
            r[(temp==i)] = self.name2color[name][0][0]
            g[(temp==i)] = self.name2color[name][0][1]
            b[(temp==i)] = self.name2color[name][0][2]

                # r[temp == l] = self.label_colours[l][0]
                # g[temp == l] = self.label_colours[l][1]
                # b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    # def encode_segmap(self, mask):
    #     # Put all void classes to zero
    #     for _voidc in self.void_classes:
    #         mask[mask == _voidc] = self.ignore_index
    #     for _validc in self.valid_classes:
    #         mask[mask == _validc] = self.class_map[_validc]
    #     return mask


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    # augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/home/n8k9/ripl/ros/data/airsim"
    dst = airsimLoader(local_path, is_transform=True) #, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
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
