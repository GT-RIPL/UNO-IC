import os
import torch
import numpy as np
import scipy.misc as m
import glob

from torch.utils import data

import matplotlib.pyplot as plt

# from ptsemseg.utils import recursive_glob
# from ptsemseg.augmentations import *


class airsimLoader(data.Dataset):
    
    name2color = {"person":  [[153, 108, 6  ]],
                  "road":    [[53,  32,  65 ], [115, 176, 195]],
                  "building":[[84,  226, 87 ]], 
                  "vehicle": [[161, 171, 27 ], [89,  121, 72 ]],
                  "tree":    [[147, 144, 233]],
                  "sidewalk":[[178, 221, 213]] }

    name2id = {"person": 1, "vehicle": 2, "road": 3, "building": 4, "sidewalk": 5, "tree": 6}
    id2name = {i:name for name,i in name2id.items()}

    splits = ['train','val']
    image_modes = ['scene','depth','segmentation']

    split_subdirs = {}
    split_subdirs['train'] = ['clear']
    split_subdirs['val'] = ['fall','fog_1','fog_2','fog_3','rain','snow']

    ignore_index = 0

    mean_rgb = {
        "airsim": [103.939, 116.779, 123.68],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
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
        self.n_classes = 7
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array(self.mean_rgb[version])
        
        self.imgs = {s:{image_mode:[] for image_mode in self.image_modes} for s in self.splits}
        for split in self.splits:
            for subdir in self.split_subdirs[split]:
                for file_path in glob.glob(os.path.join(root,'scene',subdir,"*.png")):
                    ext = file_path.replace(root+"/scene/",'')
                    if all([os.path.exists(os.path.join(root,image_mode,ext)) for image_mode in self.image_modes]):
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

        # plt.figure()
        # plt.imshow(mask)
        # plt.show()

        lbl = self.ignore_index*np.ones((img.shape[0],img.shape[1]),dtype=np.uint8)
        for i,name in self.id2name.items():
            for color in self.name2color[name]:
                lbl[(mask==color).all(-1)] = i

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

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
