import numpy as np
import torch
import torch.nn as nn


# mix up 
def mixup_data(x, y, alpha=0.4, device='cuda'):
    '''
    Compute the mixup data. Return mixed inputs, pairs of targets, and lambda
    '''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
   
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# cutmix
def cutmix_data(x, y, device='cuda'):
    """
    x: Batch of images. A 4-D torch.Tensor
    y: Batch of masks. A 3D torch.Tensor
    
    alpha == 1 is equivalent to no cutmix
    """
    indices = torch.randperm(x.size(0)).to(device)
    shuffled_x = x[indices]
    shuffled_y = y[indices]

    # lam = np.random.beta(alpha, alpha)

    image_h, image_w = x.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = np.random.uniform(image_w/5, image_w/3)
    h = np.random.uniform(image_h/5, image_h/3)
    
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    x[:, :, y0:y1, x0:x1] = shuffled_x[:, :, y0:y1, x0:x1]
    y[:, y0:y1, x0:x1] = shuffled_y[:, y0:y1, x0:x1]
    
    return x, y

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
