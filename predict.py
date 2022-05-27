import os
import argparse
import torch
from evaluate import Evaluator
from data import Test_Preprocessor
import tta
from congfigs.test_config import * 


def make_prediction(model_path, image_dir, tta_fn, mask_mode):
    model = torch.load(model_path).to(DEVICE)
    if os.path.isdir('./predict_result'): 
        import shutil 
        shutil.rmtree('./predict_result')
        print("Delete directory: predict_result/")
    os.mkdir('./predict_result')   
    print("Create directory: predict_result/")
    test_image_transform =  Test_Preprocessor(test_img_size)
    evaluator = Evaluator(model, test_image_transform, device=DEVICE, activation=activation)
    evaluator.make_prediction(image_dir, './predict_result', tta_fn, mask_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation for medical dataset!")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--tta_fn", type=check_tta_fn, default=False)
    parser.add_argument("--mask_mode", type=str)
    args = parser.parse_args()
    
    # evaluate
    make_prediction(args.model_path, args.target_dir, args.tta_fn, args.mask_mode)