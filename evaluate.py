import cv2
import os 
import torch 
from torchvision.transforms import Resize, InterpolationMode, Compose
from torchvision.transforms.functional import hflip, vflip
from data import Test_Preprocessor
# from configs.config import * 
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np


class Evaluator:
    def __init__(self, model, image_transform, device='cuda'):
        self.model = model.to(device)
        self.image_transform = image_transform
        self.device = device
    
    @torch.no_grad()
    def _predict(self, path, mask_mode='color', do_tta=True, vote_mode='any', multiscale_list=None):         
        x = self.image_transform(Image.open(path).convert('RGB'), None)[0].to(self.device)
        if do_tta: 
            X, tta_fns = self._forward_TTA(x, multiscale_list)
            mask = self._backward_TTA(self.model(X), tta_fns)
            if vote_mode == 'any': 
                mask = (mask > 0.5).any(dim=0).squeeze()
            elif vote_mode == 'soft': 
                mask = (mask.mean(dim=0).squeeze() > 0.5)
        else: 
            mask = self.model(x.unsqueeze(0)).squeeze() > 0.5

        if mask_mode ==  'color': 
            mask = torch.where(mask, 255, 0)
            self._check_range(mask, [0, 255])
        elif mask_mode == 'class': 
            mask = torch.where(mask, 1, 0)
            self._check_range(mask, [0, 1])
        return mask
    
    def evaluate(self, image_paths, label_paths, do_tta=True, vote_mode='soft', multiscale_list=None): 
        if isinstance(image_paths, str) and isinstance(label_paths, str): 
            assert os.path.isdir(image_paths) and os.path.isdir(label_paths), f"{image_paths} or {label_paths} is not a directory!"
            image_paths = sorted([os.path.join(image_paths, basename) for basename in os.listdir(image_paths)])
            label_paths = sorted([os.path.join(label_paths, basename) for basename in os.listdir(label_paths)])
            
        assert len(image_paths) == len(label_paths)
        self._check_multiscale(multiscale_list)
        
        total_score = 0 
        self.model.eval()
        for image_path, label_path in tqdm(list(zip(image_paths, label_paths))): 
            gt = torch.from_numpy(np.load(label_path)['image']).to(self.device)
            mask = self._predict(image_path, mask_mode='class', do_tta=do_tta, vote_mode=vote_mode, multiscale_list=multiscale_list)
            mask = Resize(gt.shape, InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze()
            total_score += Evaluator.dice_score(mask, gt)
        return total_score/len(image_paths)            
    
    def make_prediction(self, paths, save_dir, mask_mode='color', do_tta=True, vote_mode='soft', multiscale_list=None): 
        assert mask_mode in ('color', 'class')
        assert vote_mode in ('any', 'soft')
        assert hasattr(paths, '__iter__'), "Input Variable paths is not iterable!"
        if isinstance(paths, str): 
            assert os.path.isdir(paths), "Input variable 'path' is not a directory!"
            img_dir = paths
            paths = [os.path.join(paths, basename) for bansename in os.listdir(paths)]
            print(f"Find {len(paths)} files under {img_dir}")
        else: 
            assert os.path.isfile(path), f"'{path}' does not exist!"
        self._check_multiscale(multiscale_list)
        
        origin_size = cv2.imread(paths[0]).shape[:2]
        print(f"Use {origin_size} as output!")
        
        self.model.eval()
        for path in tqdm(paths): 
            mask = self._predict(path, mask_mode, do_tta, vote_mode, multiscale_list=multiscale_list)
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
        tta_fns = [[x, y] for x in [lambda x: x, hflip] for y in [lambda x: x, vflip]]
        if multiscale_list is not None:  
            tta_fns = [[x, y, z] for x, y in tta_fns for z in [Resize(size, interpolation=InterpolationMode.NEAREST) for size in multiscale_list]]
        tta_fns = [Compose(ele) for ele in tta_fns]
        res = [tta_fn(x) for tta_fn in tta_fns]
        return torch.stack(res, dim=0), tta_fns 
    
    def _backward_TTA(self, X, tta_fns):
        return torch.stack(
            [tta_fns[i](X.select(0, i)) for i in range(len(tta_fns))], dim=0
        )

    @staticmethod
    def dice_score(pred, gt, ep=1e-8):
        return ((2 * pred * gt).sum()/(pred.sum() + gt.sum() + ep)).item()