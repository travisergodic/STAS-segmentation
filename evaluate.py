import cv2
import os 
import torch 
import torch.nn as nn
from torchvision.transforms import Resize, InterpolationMode, Compose
from torchvision.transforms.functional import hflip, vflip
from data import Test_Preprocessor
# from configs.config import * 
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np
import ttach as tta


class Evaluator:
    def __init__(self, model, image_transform, device='cuda', activation=nn.Sigmoid()):
        self.model = model.to(device)
        self.image_transform = image_transform
        self.device = device
        self.activation = nn.Identity() if activation is None else activation
    
    @torch.no_grad()
    def _predict(self, path, mask_mode='color', do_tta=True, multiscale_list=None):         
        x = self.image_transform(Image.open(path).convert('RGB'), None)[0].to(self.device)
        
        # do test time augmentation
        if do_tta:
            # only do four flips 
            if multiscale_list is None:
                X, tta_fns = self._forward_TTA(x, None)
                mask = self._backward_TTA(self.activation(self.model(X)), tta_fns).mean(dim=0).squeeze() > 0.5
                
            # do four flips + multiscale testing 
            else: 
                origin_size = x.size()[1:] 
                X_dict, tta_fns = self._forward_TTA(x, multiscale_list)
                mask = torch.zeros(*origin_size).to(self.device)
                for key in X_dict: 
                    mask += Resize(origin_size, interpolation=InterpolationMode.NEAREST)(self._backward_TTA(self.activation(self.model(X_dict[key])), tta_fns).mean(dim=0)).squeeze()
                mask = mask/len(X_dict) > 0.5 
                
        # no test time augmentation
        else: 
            mask = self.model(x.unsqueeze(0)).squeeze() > 0.5

        if mask_mode ==  'color': 
            mask = torch.where(mask, 255, 0)
            self._check_range(mask, [0, 255])
        elif mask_mode == 'class': 
            mask = torch.where(mask, 1, 0)
            self._check_range(mask, [0, 1])
        return mask
    
    def evaluate(self, image_paths, label_paths, do_tta=True, multiscale_list=None): 
        if type(image_paths) == str and type(label_paths) == str: 
            assert os.path.isdir(image_paths) and os.path.isdir(label_paths), f"{image_paths} or {label_paths} is not a directory!"
            image_paths = sorted([os.path.join(image_paths, basename) for basename in os.listdir(image_paths)])
            label_paths = sorted([os.path.join(label_paths, basename) for basename in os.listdir(label_paths)])
            
        assert len(image_paths) == len(label_paths)
        self._check_multiscale(multiscale_list)
        
        total_score = 0 
        self.model.eval()
        for image_path, label_path in tqdm(list(zip(image_paths, label_paths))): 
            gt = torch.from_numpy(np.load(label_path)['image']).to(self.device)
            mask = self._predict(image_path, 'class', do_tta, multiscale_list)
            mask = Resize(gt.shape, InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze()
            total_score += Evaluator.dice_score(mask, gt)
        return total_score/len(image_paths)            
    
    def make_prediction(self, paths, save_dir, mask_mode='color', do_tta=True, multiscale_list=None): 
        assert mask_mode in ('color', 'class')
        assert hasattr(paths, '__iter__'), "Input Variable paths is not iterable!"
        if type(paths) == str: 
            assert os.path.isdir(paths), "Input variable 'path' is not a directory!"
            img_dir = paths
            paths = [os.path.join(paths, basename) for basename in os.listdir(paths)]
            print(f"Find {len(paths)} files under {img_dir}")
        else: 
            assert os.path.isfile(path), f"'{path}' does not exist!"
        self._check_multiscale(multiscale_list)
        
        origin_size = cv2.imread(paths[0]).shape[:2]
        print(f"Use {origin_size} as output!")
        
        self.model.eval()
        for path in tqdm(paths): 
            mask = self._predict(path, mask_mode, do_tta, multiscale_list)
            mask = Resize(origin_size, InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze()
            self._save(mask.cpu().numpy().astype(np.uint8), os.path.basename(path).split(".")[0] + ".png", save_dir)
        
    def _save(self, mask, basename, save_dir): 
        Image.fromarray(mask).save(os.path.join(save_dir, basename))
        
    def _check_range(self, mask, range_list): 
        res = torch.zeros_like(mask)
        for ele in range_list:
            res += (mask == ele).long()
        assert torch.all(res > 0)
        
    def _check_multiscale(self, multiscale_list):
        if multiscale_list is not None:
            for multiscale in multiscale_list: 
                assert isinstance(multiscale, int) and multiscale % 32 == 0
        
    def _forward_TTA(self, x, multiscale_list=None):
        tta_fns = [Compose([x, y]) for x in [lambda x: x, hflip] for y in [lambda x: x, vflip]]
        if multiscale_list is None:
            return torch.stack([tta_fn(x) for tta_fn in tta_fns], dim=0), tta_fns
        res_dict = dict()
        for scale in multiscale_list:
            res_dict[scale] = torch.stack([Resize((scale, scale), interpolation=InterpolationMode.NEAREST)(tta_fn(x)) for tta_fn in tta_fns], dim=0)
        return res_dict, tta_fns
    
    def _backward_TTA(self, X, tta_fns):
        return torch.stack(
            [tta_fns[i](X.select(0, i)) for i in range(len(tta_fns))], dim=0
        )

    @staticmethod
    def dice_score(pred, gt, ep=1e-8):
        return ((2 * pred * gt).sum()/(pred.sum() + gt.sum() + ep)).item()