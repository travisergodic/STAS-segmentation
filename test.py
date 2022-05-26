import torch
import os
import argparse
from evaluate import Evaluator
from data import Test_Preprocessor
import tta


def boolean_string(s):
    if s == 'False': 
        return False
    elif s == 'True': 
        return True   
    else:
        raise ValueError('Not a valid boolean string')

def check_tta_fn(s): 
    return {
        "d4": tta.aliases.d4_transform()
    }.get(s, False)

def evaluate_all(model, test_image_path_list, test_label_path_list):
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
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--label_dir", type=str)
    parser.add_argument("--tta_fn", type=check_tta_fn)
    args = parser.parse_args()

    # read val.txt file
    with open("./val.txt", "r") as f: 
        test_image_path_list = [
            os.path.join(args.image_dir, line.strip()) for line in f.readlines()
        ]
        
        test_label_path_list = [
            os.path.join(
                args.label_dir, 
                "label_" + os.path.basename(image_path).split(".")[0] + ".npz"
            ) for image_path in test_image_path_list
        ]

    print(f"Read 'val.txt' file successfully! {len(test_image_path_list)} evaluation images!")
    # build model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.model_path).to(DEVICE)
    evaluate_all(model, test_image_path_list, test_label_path_list)