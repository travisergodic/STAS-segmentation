import os 
import torch 
from torchvision.transforms import Resize, InterpolationMode
from torchvision.transforms.functional import hflip, vflip
from data import Test_Preprocessor
from config import * 
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
    def _predict(self, path, mask_mode='color', do_tta=True, vote_mode='any'):         
        x = self.image_transform(Image.open(path).convert('RGB'), None)[0].to(self.device)
        if do_tta: 
            X, tta_fns = self._forward_TTA(x)
            mask = self._backward_TTA(model(X), tta_fns)
            if vote_mode == 'any': 
                mask = (mask > 0.5).any(dim=0).squeeze()
            elif vote_mode == 'soft': 
                mask = (mask.mean(dim=0).squeeze() > 0.5)
        else: 
            mask = model(x.unsqueeze(0)).squeeze() > 0.5

        if mask_mode ==  'color': 
            mask = torch.where(mask, 255, 0)
            self._check_range(mask, [0, 255])
        elif mask_mode == 'class': 
            mask = torch.where(mask, 1, 0)
            self._check_range(mask, [0, 1])
        return mask
    
    def evaluate(self, image_paths, label_paths, do_tta=True, vote_mode='any'): 
        assert len(image_paths) == len(label_paths)
        total_score = 0 
        self.model.eval()
        for image_path, label_path in tqdm(list(zip(image_paths, label_paths))): 
            gt = torch.from_numpy(np.load(label_path)['image']).to(self.device)
            mask = self._predict(image_path, mask_mode='class', do_tta=do_tta, vote_mode=vote_mode)
            mask = Resize(gt.shape, InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze()
            total_score += Evaluator.dice_score(mask, gt)
        return total_score/len(image_paths)            
    
    def test(self, paths, save_dir, mask_mode='color', do_tta=True, vote_mode='any'): 
        assert mask_mode in ('color', 'class')
        assert vote_mode in ('any', 'soft')
        assert hasattr(paths, '__iter__'), "Input Variable paths is not iterable!"
        for path in paths: 
            assert os.path.isfile(path), f"'{path}' does not exist!"
        
        self.model.eval()
        for path in tqdm(paths): 
            mask = self._predict(path, mask_mode, do_tta, vote_mode)
            self._save(mask.cpu().numpy().astype(np.uint8), path, save_dir)
        
    def _save(self, mask, path, save_dir): 
        save_path = os.path.join(save_dir, os.path.basename(path).split(".")[0] + ".png")
        Image.fromarray(mask).save(save_path)
        
    def _check_range(self, mask, range_list): 
        res = torch.zeros_like(mask)
        for ele in range_list:
            res += (mask == ele).long()
        assert torch.all(res > 0)
        
    def _forward_TTA(self, x):
        res = []
        tta_fns = [(x, y) for x in [lambda x: x, hflip] for y in [lambda x: x, vflip]]
        for aug_pair in tta_fns:
            res.append(aug_pair[1](aug_pair[0](x)))
        return torch.stack(res, dim=0), tta_fns 
    
    def _backward_TTA(self, X, tta_fns): 
        res = []
        for index in range(len(tta_fns)):
            tta_fn = tta_fns[index]
            res.append(tta_fn[1](tta_fn[0](X.select(0, index))))
        return torch.stack(res, dim=0)

    @staticmethod
    def dice_score(pred, gt, ep=1e-8):
        return ((2 * pred * gt).sum()/(pred.sum() + gt.sum() + ep)).item()      