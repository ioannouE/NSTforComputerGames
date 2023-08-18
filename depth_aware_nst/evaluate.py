import argparse
import os
import sys
import time
import re

import numpy as np
import torch

import utils
import cv2

import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *
from PIL import Image

import fast_neural_style
# import calculate_depth_loss

# from diffimg import diff

import glob
from path import Path


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


# find the style
STYLES = ['composition_vii', 'feathers', 'fire', 'mosaic', 'starry_night', 'the_scream', 'wave'] # , 'the_muse']
COMPOSITE_IMAGE = 'composite_image.png'

INPUT_DIRS = ['../../unity_games_test/game_1_demo_scene_1', '../../unity_games_test/game_1_demo_scene_2', '../../unity_games_test/game_1_demo_scene_3',
                  '../../unity_games_test/game_2_fontaine_scene_1', '../../unity_games_test/game_2_fontaine_scene_2', '../../unity_games_test/game_2_fontaine_scene_3',
                  '../../unity_games_test/game_3_dead_scene_1', '../../unity_games_test/game_3_dead_scene_2', '../../unity_games_test/game_3_dead_scene_3',
                  '../../unity_games_test/game_4_seed_scene_1', '../../unity_games_test/game_4_seed_scene_2', '../../unity_games_test/game_4_seed_scene_3']

# INPUT_DIRS = ['../../unity_games_test/game_2_fontaine_scene_1']
    #############################################################################

def main():
    
    parser = argparse.ArgumentParser(description='parser for evaluating a model')
    parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. File should end in .pth")
    parser.add_argument("--cuda", type=int, required=False, default=1,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--depth-loss", type=int, default=0,
                                  help="set it to 1 to calculate average depth loss, default is 1")
    parser.add_argument("--content-path", type=str, required=False,
                                  help="path to content images")
    
    
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: ", torch.cuda.get_device_name(0))

    for input_dir in INPUT_DIRS:

        # create directory
        new_dir = str(args.model).replace('saved_models/', '').replace('.pth','').replace('to_onnx/', '')
        print(new_dir)
        parent_dir = "images/output/"
        path = os.path.join(parent_dir, new_dir)
        scene = input_dir.replace('../../unity_games_test/','')
        path = os.path.join(path, scene)

        if not os.path.exists(path):
            os.makedirs(path)
    # model_type = "DPT_Large"

        print("SCENE: ", scene)

    # midas = torch.hub.load("intel-isl/MiDaS", model_type)
    # midas.to(device)
    # midas.eval()

    # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    # if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    #     transform = midas_transforms.dpt_transform
    # else:
    #     transform = midas_transforms.small_transform


   
        # print(args.content_path)
        # print(Path(args.content_path).stem[:-1])
        # iterate over all content images and produce the sytlised output
        for content_image in glob.glob(input_dir + '/*.jpg'): 
            # print('Stylising ', content_image)
            fast_neural_style.stylize_p(content_image, args.model, args.cuda, input_dir.replace(scene,''))

        print('Stylised images saved in ', path)

        # stylised images
        # for img in glob.glob(path + '/*'):
        #     stylised_images.append(img)

        # # check if the model has been trained with depth loss
        # if args.depth_aware:
        #     # find the stylised result with no depth
        #     model_style = ''
        #     for style in STYLES:
        #         if style in path:
        #             model_style = style
            
        #     if model_style == '': 
        #         print('Model name is not valid. It should contain one of the style images.')
        #         sys.exit(1)

        #     stylised_images_no_depth = []
        #     for img in glob.glob('images/output/' + model_style + '/*'):
        #         print('stylised image no depth: ', img)
        #         if not COMPOSITE_IMAGE in img:
        #             stylised_images_no_depth.append(img)

        #     # compute midas depth loss for stylised and stylised no depth
        #     rows = []
        #     for cont_img, stylised_img_no_depth, stylised_img in zip(content_images, stylised_images_no_depth, stylised_images):
        #         # generate diff image
        #         diff_img_name = path + '/' + model_style + '_diff.png'
        #         r = diff(stylised_img_no_depth, stylised_img, diff_img_file=diff_img_name)
                
        #         c_img = cv2.imread(cont_img)
        #         s_img_no_depth = cv2.imread(stylised_img_no_depth)
        #         s_img = cv2.imread(stylised_img)
        #         diff_img = cv2.imread(diff_img_name)

        #         # compute depth images
        #         c_img_depth_img = compute_depth_map_image(args, c_img, midas, transform, device)
        #         s_img_no_depth_depth_img = compute_depth_map_image(args, s_img_no_depth, midas, transform, device)
        #         s_img_depth_img = compute_depth_map_image(args, s_img, midas, transform, device)
                
        #         c_img = cv2.copyMakeBorder(c_img,10,10,10,20,cv2.BORDER_CONSTANT,value=[255,255,255])
        #         s_img_no_depth = cv2.copyMakeBorder(s_img_no_depth,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
        #         s_img = cv2.copyMakeBorder(s_img,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
        #         diff_img = cv2.copyMakeBorder(diff_img,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])            

        #         rows.append(hconcat_resize_min([c_img, c_img_depth_img, s_img_no_depth, s_img, s_img_no_depth_depth_img, s_img_depth_img, diff_img]))

        
        #     out_img = vconcat_resize_min(rows)
        #     cv2.imwrite(path + '/' + COMPOSITE_IMAGE, out_img)

    if args.depth_loss:
        average_mse_depth_loss = average_mse_depth(content_images, stylised_images, midas, transform, device)
        print('Average depth loss: ', average_mse_depth_loss)



# compute average mse depth loss for each content image and each stylised image
def average_mse_depth(content_images, stylised_images, midas, transform, device):

    sum = 0
    for c_img, s_img in zip(content_images, stylised_images):
        if not COMPOSITE_IMAGE in s_img:
            c_img = cv2.imread(c_img)
            s_img = cv2.imread(s_img)
            loss, (c_img_depth, s_img_depth) = calculate_depth_loss.calculate_loss(c_img, s_img, midas, transform, device)
            sum += loss

    return (sum / len(stylised_images))



# function to produce depth map from a given input image
def compute_depth_map_image(args, input_image, midas, transform, device):

    input_batch = transform(input_image).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction_img = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=input_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction_img.cpu().numpy()

    formatted = (output * 255 / np.max(output)).astype('uint8')
    img = Image.fromarray(formatted)

    output_img = img.convert('RGB')
    return np.array(output_img)

    


if __name__ == "__main__":
    main()