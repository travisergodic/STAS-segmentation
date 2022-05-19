import torch
from torch import optim

# config
train_image_dir = './data/Train_Images/'
label_dir = './data/Annotations/'
seed = 14

# preprocess
ann_suffix = '.npz'
train_img_size = (384, 384)
test_img_size = (384, 384)
h_flip_p=0.5
v_flip_p=0.5
affine_p=0.

# dataloader
train_ratio = 0.85
train_batch_size = 16
test_batch_size = 32

# train config 
num_epoch = 100
decay_fn = lambda n: 1
is_sam = False 
optim_cls = optim.Adam
optim_dict = {
    'lr': 1e-4, 
    # 'weight_decay': 1e-2
}

# device 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

