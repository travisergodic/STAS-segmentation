import ttach as tta
import torch
import torch.nn as nn 

test_img_size = (512, 512)
tta_fn = tta.aliases.flip_transform()
activation = nn.Sigmoid()
image_dir = '/content/STAS-segmentation/Train_Images/'
label_dir = '/content/STAS-segmentation/Annotations/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'