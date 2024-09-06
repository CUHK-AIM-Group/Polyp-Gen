from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
import torch
import pandas as pd
from PIL import Image
import numpy as np
import os
import argparse


def main(args):
    save_path = args.save_path
    checkpoint_path = args.checkpoint_path
    test_csv = args.test_file
    data_path = args.dataset_path

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        checkpoint_path,
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    if not os.path.exists(save_path + "/0"):
        os.makedirs(save_path + "/0")
    if not os.path.exists(save_path + "/1"):
        os.makedirs(save_path + "/1")

    pipe = pipe.to('cuda')

    for index, row in test_csv.iterrows():
        image_path = data_path + "/" + row['image']
        mask_path = data_path + "/" + row['seg']
        image = Image.open(image_path)
        mask_image = Image.open(mask_path)

        #Enlarge the range of Mask
        mask = np.array(mask_image)
        rows, cols = np.where(mask == 255)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        #Enlarge the range of Mask
        min_row = max(0, min_row - 15)
        max_row = min(mask.shape[0], max_row + 15)
        min_col = max(0, min_col - 15)
        max_col = min(mask.shape[1], max_col + 15)
        mask = np.zeros_like(mask)
        mask[min_row:max_row, min_col:max_col] = 255
        mask_image = Image.fromarray(mask, "L")

        if row['label'] == 0:
            prompt = "Normal"
        else:
            prompt = "Polyp"

        gen_image = pipe(prompt=prompt, image=image, mask_image=mask_image,
            width=image.size[0], height=image.size[1], num_inference_steps=50,
                ).images[0]
        
        if index < 1000:
            gen_image.save(save_path + "/0/img_" + str(index) + ".jpg")
        else:
            gen_image.save(save_path + "/1/img_" + str(index) + ".jpg")



if __name__ == "__main__":
    #add Args
    parser = argparse.ArgumentParser(description='Sample inpainting')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the checkpoints')
    parser.add_argument('--data_path', type=str, required=True, help='Path of Dataset')
    parser.add_argument('--test_file', type=str, required=True, help='Test *.csv file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the inpainted images')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the UNet checkpoint')
    args = parser.parse_args()
    main(args)
