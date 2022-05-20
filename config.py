import torch
from torch import optim
import segmentation_models_pytorch as smp
from vision_transformer import SwinUnet
from transformers import MaskFormerModel

# device 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# config
train_image_dir = './data/Train_Images/'
label_dir = './data/Annotations/'
seed = 14

# preprocess
ann_suffix = '.npz'
img_suffix = '.jpg'
train_img_size = (384, 384)
test_img_size = (384, 384)
h_flip_p=0.5
v_flip_p=0.5
affine_p=0.

# dataloader
train_ratio = 0.85
train_batch_size = 16
test_batch_size = 32
num_workers = 4

# train config 
num_epoch = 100
decay_fn = lambda n: 1
is_sam = False 
optim_cls = optim.Adam
optim_dict = {
    'lr': 1e-4, 
    # 'weight_decay': 1e-2
}

## model 
model = smp.Unet(
    encoder_name="tu-tf_efficientnetv2_l_in21k",     
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=1,
    decoder_attention_type='scse'
    ).to(DEVICE)

# 384, 480
# tu-tf_efficientnetv2_l_in21k
# model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-base-ade")
# model = SwinUnet().to(DEVICE)

## save
save_config = {
    "path": './resunet.pt', 
    "freq": 1
}

## loss function 
loss_fn = smp.utils.losses.DiceLoss(activation='sigmoid')

# lm.GDiceLossV2(nn.Softmax(dim=1), do_bg=False, smooth=1e-5)
# lm.IoULoss(nn.Softmax(dim=1), batch_dice=True, do_bg=False, smooth=1e-6, square=True)
# lm.SoftDiceLoss(nn.Softmax(dim=1), batch_dice=True, do_bg=False, smooth=1e-6, square=True)
# lm.BceLoss()
# lm.BceDiceLoss()

## metric
metric_dict = {
    'IOU': smp.utils.metrics.IoU(threshold=0.5),
    'Accuracy': smp.utils.metrics.Accuracy(), 
    'DICE': smp.utils.losses.DiceLoss(activation='sigmoid')
}