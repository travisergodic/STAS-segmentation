import os
import time
import numpy as np 
import glob
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from trainer import Trainer
from data import StasDataset, Train_Preprocessor, Test_Preprocessor
from batch_sampler import BatchSampler,RandomSampler
from configs.config import * 
from evaluate import Evaluator
from hooks import *


def boolean_string(s):
    if s == 'False': 
        return False
    elif s == 'True': 
        return True   
    else:
        raise ValueError('Not a valid boolean string')

def check_mode(mode): 
    if mode not in ("train", "evaluate", "make_prediction"): 
        raise ValueError('Not a valid mode')
    return mode

def decode_multiscale(multiscale): 
    if multiscale is not None: 
        return [int(ele.strip()) for ele in multiscale.split(",")]
    else: 
        return None
     
        
def train(): 
    # do train test split & create val.txt file
    if not os.path.isfile("./val.txt"):         
        np.random.seed(seed)
        image_path_list = sorted(glob.glob(train_image_dir + "*" + img_suffix))
        np.random.shuffle(image_path_list)
        assert len(image_path_list) > 0
        split_index = int(len(image_path_list) * train_ratio)
        train_path_list = image_path_list[:split_index]
        test_path_list = image_path_list[split_index:]

        with open("./val.txt", "w") as f: 
            res = "\n".join([os.path.basename(test_path) for test_path in test_path_list])
            f.write(res)
            print("Create 'val.txt' file successfully!")
            
    # read val.txt file & create corresponding trainig and validation set 
    else: 
        print("'val.txt' file already exists!")
        with open("./val.txt", "r") as f: 
            image_path_list = [os.path.normpath(path) for path in glob.glob(train_image_dir + "*" + img_suffix)]
            assert len(image_path_list) > 0
            test_path_list = [os.path.normpath(os.path.join(train_image_dir, line.strip())) for line in f.readlines()]
            train_path_list = [image_path for image_path in image_path_list if image_path not in test_path_list]
        print("Read 'val.txt' file successfully!")
            
    
    print(f"Training set: {len(train_path_list)} images. \nValidation set: {len(test_path_list)} images. \n")
    
    # preprocesor 
    train_image_transform = Train_Preprocessor(None if do_multiscale else train_img_size,
                                         h_flip_p=h_flip_p,
                                         v_flip_p=v_flip_p)
    test_image_transform = Test_Preprocessor(test_img_size)
    
    # dataset
    train_dataset = StasDataset(train_path_list, label_dir, train_image_transform, ann_suffix)
    test_dataset = StasDataset(test_path_list, label_dir, test_image_transform, ann_suffix)
    
    # batchsampler & dataloader 
    if do_multiscale:
        batch_sampler = BatchSampler(RandomSampler(train_dataset),
                                     batch_size=train_batch_size,
                                     drop_last=False,
                                     multiscale_step=multiscale_step,
                                     img_sizes=multiscale_list)
        shuffle = False
    else: 
        batch_sampler = None
        shuffle = True
        
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        shuffle=shuffle
    )
   
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        num_workers=num_workers)

    # create model
    if checkpoint_path is not None: 
        model = torch.load(checkpoint_path).to(DEVICE)
        print(f'Load model from {checkpoint_path} successfully!')
    else:
        model = model_cls(**model_config).to(DEVICE)
    
    # train_model
    start = time.time()
    
    ## get iter_hook_cls
    iter_hook_cls = {
        "sam": SAM_Iter_Hook, 
        "mixup": Mixup_Iter_Hook, 
        "cutmix": Cutmix_Iter_Hook, 
        "half_cutmix": Half_Cutmix_Iter_Hook,
        "cutout": Cutout_Iter_Hook,
        "normal": Normal_Iter_Hook
    }.get(regularization_option, Normal_Iter_Hook)
    
    print(f"Use iter hook of type <class {iter_hook_cls.__name__}> during training!")
    train_pipeline = Trainer(optim_cls, decay_fn, loss_fn, metric_dict, iter_hook_cls(), DEVICE, **optim_dict)
    train_pipeline.fit(model, train_dataloader, test_dataloader, num_epoch, save_config)
    print(f"Training takes {time.time() - start} seconds!")
    
    
def evaluate_all(model, test_image_path_list, test_label_path_list):
    test_image_transform =  Test_Preprocessor(test_img_size)
    evaluator = Evaluator(model, test_image_transform, device='cuda', activation=activation)
    # normal
    score = evaluator.evaluate(test_image_path_list, test_label_path_list, False)
    print(f"No TTA: {score} (Dice score).")
    # tta
    score = evaluator.evaluate(test_image_path_list, test_label_path_list, tta_fn)
    print(f"With TTA: {score} (Dice score).")    
    
    
def make_prediction(model, image_dir, do_tta, mask_mode):
    if os.path.isdir('./predict_result'): 
        import shutil 
        shutil.rmtree('./predict_result')
        print("Delete directory: predict_result/")
    os.mkdir('./predict_result')   
    print("Create directory: predict_result/")
    test_image_transform =  Test_Preprocessor(test_img_size)
    evaluator = Evaluator(model, test_image_transform, device='cuda', activation=activation)
    evaluator.make_prediction(image_dir, './predict_result', tta_fn if do_tta else False, mask_mode)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation for medical dataset!")
    parser.add_argument("--mode", type=check_mode, default='train')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--do_tta", type=boolean_string)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--mask_mode", type=str)
    args = parser.parse_args()
    
    # input mode
    if args.mode == "train": 
        train()
    
    # input mode, model_path
    elif args.mode == "evaluate":
        # read val.txt file
        with open("./val.txt", "r") as f: 
            test_image_path_list = [
                os.path.join(train_image_dir, line.strip()) for line in f.readlines()
            ]
            
            test_label_path_list = [
                os.path.join(
                    label_dir, 
                    "label_" + os.path.basename(image_path).split(".")[0] + ".npz"
                ) for image_path in test_image_path_list
            ]
            
            print(f"Read 'val.txt' file successfully! {len(test_image_path_list)} evaluation images!")
        
        # build model
        model = torch.load(args.model_path).to(DEVICE)
        # evaluate
        evaluate_all(model, test_image_path_list, test_label_path_list)
    
    # input mode, model_path, target_dir, mask_mode, do_tta, vote_mode, multiscale_list
    elif args.mode == "make_prediction": 
        # build model
        model = torch.load(args.model_path).to(DEVICE)
        # evaluate
        make_prediction(model, args.target_dir, args.do_tta, args.mask_mode)