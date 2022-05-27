import torch
import os
import argparse
from evaluate import Evaluator
from data import Test_Preprocessor
from configs.test_config import *


def boolean_string(s):
    if s == 'False': 
        return False
    elif s == 'True': 
        return True   
    else:
        raise ValueError('Not a valid boolean string')

def evaluate_all(model_path, test_image_path_list, test_label_path_list):
    model = torch.load(model_path).to(DEVICE)
    test_image_transform =  Test_Preprocessor(test_img_size)
    evaluator = Evaluator(model, test_image_transform, device='cuda', activation=activation)
    # normal
    score = evaluator.evaluate(test_image_path_list, test_label_path_list, False)
    print(f"No TTA: {score} (Dice score).")
    # tta
    score = evaluator.evaluate(test_image_path_list, test_label_path_list, tta_fn)
    print(f"With TTA: {score} (Dice score).")   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation for medical dataset!")
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    # read val.txt file
    with open("./val.txt", "r") as f: 
        test_image_path_list = [
            os.path.join(image_dir, line.strip()) for line in f.readlines()
        ]
        
        test_label_path_list = [
            os.path.join(
                label_dir, 
                "label_" + os.path.basename(image_path).split(".")[0] + ".npz"
            ) for image_path in test_image_path_list
        ]

    print(f"Read 'val.txt' file successfully! {len(test_image_path_list)} evaluation images!")
    # build model
    
    evaluate_all(args.model_path, test_image_path_list, test_label_path_list)