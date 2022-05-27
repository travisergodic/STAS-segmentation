import ttach as tta
import torch
import torch.nn as nn 

test_img_size = (448, 448)
tta_fn = tta.aliases.d4_transform()
activation = nn.Sigmoid()
image_dir = '/content/STAS-segmentation/Train_Images/'
label_dir = '/content/STAS-segmentation/Annotations/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'