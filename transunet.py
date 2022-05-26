import os
import shutil
import numpy as np 

def build_transunet(name, img_size, num_classes, pretrained=True):
    from vit_seg_modeling import VisionTransformer as ViT_seg
    from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    config_vit = CONFIGS_ViT_seg[name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3
    
    if name.find('R50') != -1:
        vit_patches_size = int(name[-2:])
        assert img_size % vit_patches_size == 0
        config_vit.patches.grid = (int(img_size/vit_patches_size), int(img_size/vit_patches_size))

    # build model
    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)

    # dwonload pretrained weights
    if pretrained:
        import wget
        url_dict = {
            "R50-ViT-B_16": "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz",
            "ViT-B_16": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz",
            "ViT-B_32": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz", 
            "ViT-L_16": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz",
            "ViT-L_32": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz",
            "R50-ViT-L_16": "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-L_16.npz"
        }
        url = url_dict.get(name, False)
        assert url 
        if url and (not os.path.isfile(os.path.join('./pretrain_models', url.split('/')[-1]))):  
            print("Download pretrained weight!")
            filename = wget.download(url)
            if not os.path.isdir('./pretrain_models'):
                os.mkdir('./pretrain_models')
            shutil.move(os.path.join(os.getcwd(), filename), './pretrain_models') 

    # load pretrained weights
    model.load_from(weights=np.load(os.path.join('./pretrain_models', url.split('/')[-1])))
    return model