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
from hooks import SAM_Iter_Hook, Mixup_Iter_Hook, Cutmix_Iter_Hook, Normal_Iter_Hook


np.random.seed(seed)
image_path_list = sorted(glob.glob(train_image_dir + "*" + img_suffix))
np.random.shuffle(image_path_list)
assert len(image_path_list) > 0
split_index = int(len(image_path_list) * train_ratio)
train_path_list = image_path_list[:split_index]
test_path_list = image_path_list[split_index:]


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
    image_path_list = sorted(glob.glob(train_image_dir + "*" + img_suffix))
    assert len(image_path_list) > 0
    
    split_index = int(len(image_path_list) * train_ratio)
    train_path_list = image_path_list[:split_index]
    test_path_list = image_path_list[split_index:]
    
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
        model = torch.load(checkpoint_path).to(device)
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
        "normal": Normal_Iter_Hook
    }.get(regularization_option, Normal_Iter_Hook)
    
    print(f"Use iter hook of type <class {iter_hook_cls.__name__}> during training!")
    train_pipeline = Trainer(optim_cls, decay_fn, loss_fn, metric_dict, iter_hook_cls(), DEVICE, **optim_dict)
    train_pipeline.fit(model, train_dataloader, test_dataloader, num_epoch, save_config)
    print(f"Training takes {time.time() - start} seconds!")
    
    
def evaluate_all(model, test_image_path_list, test_label_path_list, multiscale_list):
    test_image_transform =  Test_Preprocessor(test_img_size)
    evaluator = Evaluator(model, test_image_transform, device='cuda')
    # no aug
    score = evaluator.evaluate(test_image_path_list, test_label_path_list, do_tta=False)
    print(f"No TTA: {score} (Dice score).")
    # only flip 
    evaluator = Evaluator(model, test_image_transform, device='cuda')
    score = evaluator.evaluate(test_image_path_list, test_label_path_list, do_tta=True, multiscale_list=None)
    print(f"Only Flip TTA soft voting: {score} (Dice score).")
    # flip + multiscale
    
    if multiscale_list is not None:
        test_image_transform =  Test_Preprocessor(None)
        score = evaluator.evaluate(test_image_path_list, test_label_path_list, do_tta=True, multiscale_list=multiscale_list)
        print(f"Flip + Multiscale TTA soft voting: {score} (Dice score).")
    
    
def make_prediction(model, image_dir, mask_mode, do_tta, multiscale_list):
    if os.path.isdir('./predict_result'): 
        import shutil 
        shutil.rmtree('./predict_result')
        print("Delete directory: predict_result/")
    os.mkdir('./predict_result')   
    print("Create directory: predict_result/")
    test_image_transform =  Test_Preprocessor(None if (multiscale_list is not None and do_tta) else test_img_size)
    evaluator = Evaluator(model, test_image_transform, device='cuda')
    evaluator.make_prediction(image_dir, './predict_result', mask_mode, do_tta, multiscale_list)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation for medical dataset!")
    parser.add_argument("--mode", type=check_mode, default='train')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--do_tta", type=boolean_string)
    parser.add_argument("--multiscale", type=decode_multiscale, default=None)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--mask_mode", type=str)
    args = parser.parse_args()
    
    # input mode
    if args.mode == "train": 
        train()
        
    # input mode, model_path, multiscale_list
    elif args.mode == "evaluate":
        test_image_path_list = test_path_list
        test_label_path_list = [os.path.join(label_dir, "label_" + os.path.basename(image_path).split(".")[0] + ".npz") for image_path in test_image_path_list]
        
        model = torch.load(args.model_path).to(DEVICE)
        evaluate_all(model, test_image_path_list, test_label_path_list, args.multiscale_list)
    
    # input mode, model_path, target_dir, mask_mode, do_tta, vote_mode, multiscale_list
    elif args.mode == "make_prediction": 
        model = torch.load(args.model_path).to(DEVICE)
        make_prediction(model, args.target_dir, args.mask_mode, args.do_tta, args.multiscale_list)