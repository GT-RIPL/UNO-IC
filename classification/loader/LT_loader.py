import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from collections import Counter


# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, image_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size = image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean = rgb_mean, std=rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop([image_size,image_size]),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop([image_size,image_size]),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]

# Dataset
class LT_Loader(Dataset):
    
    def __init__(self, root, ann_file, phase='train', image_size =224):
        self.img_path = []
        self.labels = []
        self.n_classes = 365
        mean =  [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = get_data_transform(phase, mean, std, image_size)
        with open(ann_file) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)

    def get_cls_num_list(self):
        cls_num_list = []
        cls_num_dict = Counter(self.labels)
        for k in range(self.n_classes):
            cls_num_list.append(cls_num_dict[k])
        return cls_num_list

    def get_balanced_weight(self):
        cls_num_list = self.get_cls_num_list()
        
        beta = (np.sum(cls_num_list)-1)/np.sum(cls_num_list)
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        return per_cls_weights
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        sample = Image.open(path).convert('RGB')
        # import ipdb;ipdb.set_trace()
        # if self.transform is not None:
        sample = self.transform(sample)

        return sample, label

    