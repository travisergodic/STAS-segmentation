import os
import random
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import patchify
from configs.config import * 

class StasDataset(Dataset):
    def __init__(self, image_path_list, label_dir, image_transform=None, ann_suffix=ann_suffix):
        self.image_path_list = image_path_list
        self.label_dir = label_dir
        self.image_transform = image_transform 
        self.ann_suffix = ann_suffix
    
    def __getitem__(self, index):
        image = Image.open(self.image_path_list[index]).convert('RGB')
        
        if self.ann_suffix == '.png':
            label_path = os.path.join(self.label_dir, os.path.basename(self.image_path_list[index]).split(".")[0] + self.ann_suffix)
            label = torch.from_numpy(np.array(Image.open(label_path))).unsqueeze(dim=0)
        elif self.ann_suffix == '.npz': 
            label_path = os.path.join(self.label_dir, 'label_' + os.path.basename(self.image_path_list[index]).split(".")[0] + self.ann_suffix)
            label = torch.from_numpy(np.load(label_path)['image']).unsqueeze(dim=0)

        if self.image_transform is not None:
            image, label = self.image_transform(image, label)
        return image, label

    def __len__(self):
        return len(self.image_path_list)


class Train_Preprocessor(nn.Module): 
    def __init__(self, img_size, h_flip_p=0.5, v_flip_p=0.5):
        super().__init__()
        self.img_size = img_size
        self.resize = transforms.Resize(self.img_size, interpolation=InterpolationMode.NEAREST) 
        self.jitter = transforms.ColorJitter(0.25, 0.25, 0.25)
        self.blur = transforms.GaussianBlur((1, 3))
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p
        self.preprocess = transforms.Compose(
            [
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
    
    @torch.no_grad()
    def forward(self, img, label): 
        # random crop
        # i, j = random.randint(0, 470), random.randint(0, 857)
        # img = F.crop(img, i, j, 471, 858)
        # label = F.crop(label, i, j, 471, 858)

        # resize & color transform 
        img = self.blur(self.jitter(self.resize(img)))
        label = self.resize(label)

        # Random horizontal flipping
        if random.random() < self.h_flip_p:
            img = F.hflip(img)
            label = F.hflip(label)

        # Random vertical flipping
        if random.random() < self.v_flip_p:
            img = F.vflip(img)
            label = F.vflip(label)        

        # random affine
        # if random.random() < self.affine_p: 
        #     affine_param = transforms.RandomAffine.get_params(
        #         degrees = [-10, 10], translate = [0.1,0.1],  
        #         img_size = [512, 512], scale_ranges = [1, 1.3], 
        #         shears = [2,2])

        #     img = F.affine(img, 
        #                    affine_param[0], affine_param[1],
        #                    affine_param[2], affine_param[3])

        #     label = F.affine(label, 
        #                      affine_param[0], affine_param[1],
        #                      affine_param[2], affine_param[3])

        # random_resize_param = transforms.RandomResizedCrop.get_params(
        #     img,
        #     scale=(0.90, 1.0),
        #     ratio=(0.90, 1.3333333333333333)
        # )

        # img = F.resized_crop(img, 
        #                      random_resize_param[0], random_resize_param[1], 
        #                      random_resize_param[2], random_resize_param[3], 
        #                      self.img_size, InterpolationMode.NEAREST)

        # label = F.resized_crop(label, 
        #                      random_resize_param[0], random_resize_param[1], 
        #                      random_resize_param[2], random_resize_param[3], 
        #                      self.img_size, InterpolationMode.NEAREST)
        return self.preprocess(img), label


class Test_Preprocessor(nn.Module): 
    def __init__(self, img_size, patches=None):
        super().__init__()
        self.img_size = img_size
        self.patches = patches

        if self.patches is not None: 
            self.resize = transforms.Resize((img_size[0] * patches[0], img_size[1] * patches[1]), interpolation=InterpolationMode.NEAREST)
        else: 
            self.resize = transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST)
        
        self.preprocess = transforms.Compose(
            [ 
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

    @torch.no_grad()
    def forward(self, img, label): 
        if self.patches is None: 
            return self.preprocess(self.resize(img)), None if label is None else self.resize(label)
        img, label = self.preprocess(self.resize(img)).numpy(), None if label is None else self.resize(label).numpy()
        img_patches = patchify.patchify(img, (3, self.img_size[0], self.img_size[1]), step=(1, self.img_size[0], self.img_size[1]))
        
        label_patches = None if label is None else patchify.patchify(label, (1, self.img_size[0], self.img_size[1]), step=(1, self.img_size[0], self.img_size[1]))
        return img_patches, label_patches
