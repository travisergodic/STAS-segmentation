import numpy as np
import torch


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

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)