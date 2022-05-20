import os
import time
import numpy as np 
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from trainer import Trainer
from data import StasDataset, Train_Preprocessor, Test_Preprocessor
from configs.config import *


if __name__ == "__main__":
    np.random.seed(seed)
    image_path_list = sorted(glob.glob(train_image_dir + "*" + img_suffix))
    assert len(image_path_list) > 0
    
    split_index = int(len(image_path_list) * train_ratio)
    train_path_list = image_path_list[:split_index]
    test_path_list = image_path_list[split_index:]

    
    # dataset & dataloader
    train_data = StasDataset(train_path_list, label_dir, image_transform=Train_Preprocessor(train_img_size, 
                                                                                            h_flip_p=h_flip_p, 
                                                                                            v_flip_p=v_flip_p))
    test_data = StasDataset(test_path_list, label_dir, image_transform=Test_Preprocessor(test_img_size))

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, num_workers=num_workers)

    # train_model
    start = time.time()
    train_pipeline = Trainer()
    train_pipeline.compile(optim_cls, decay_fn, loss_fn, metric_dict, is_sam, DEVICE, **optim_dict)
    train_pipeline.fit(model, train_dataloader, test_dataloader, num_epoch, save_config)
    print(f"Training takes {time.time() - start} seconds!")