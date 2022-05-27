import ttach as tta
import torch
import torch.nn as nn 

test_img_size = (384, 384)
tta_fn = tta.aliases.d4_transform()
activation = nn.Sigmoid()
image_dir = './data/Train_Images/'
label_dir = './data/Annotations/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'