import argparse
import os
import sys

from skimage import io, transform
from torch.utils.data import DataLoader
import torch
torch.cuda.empty_cache()

import gc
gc.collect()

import re
import numpy as np
from scipy import misc
from PIL import Image
from torch.autograd import Variable
import glob
from torchvision import transforms
import matplotlib as plt
import cv2

import lpips

from vgg import Vgg16
import utils

# https://github.com/safwankdb/ReCoNet-PyTorch/blob/master/testwarp.py

def get_subdirectories(directory_path):
    subdirectories = []
    for entry in os.scandir(directory_path):
        if entry.is_dir():
            subdirectories.append(entry.name)
    return subdirectories



def main():
    
    parser = argparse.ArgumentParser(description='parser for evaluating a model')
    parser.add_argument("--frames", type=str, required=True,
                        help="folder that contains the images")
    # parser.add_argument("--model", type=str, required=True,
    #                     help="the name of the model")
    # parser.add_argument("--opticflow", type=str, required=True,
    #                              help="folder that contains the optic flow")
    parser.add_argument("--cuda", type=int, default=1, required=False,
                                  help="use cuda")
    parser.add_argument("--image-size", type=int, default=360,
                                  help="the image size")
    parser.add_argument("--style-image", type=str, required=True,
                                  help="the style image")
    
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

     # set up midas
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: ", torch.cuda.get_device_name(0))
    print("Running on ", device)

    mse_loss = torch.nn.MSELoss()
    lpips_sim = lpips.LPIPS(net='squeeze').to(device)
    vgg = Vgg16(requires_grad=False).to(device)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image)
    style = style_transform(style).to(device)
    style = style.repeat(1, 1, 1, 1).to(device)
    
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # INPUT_DIRS = ['game/input/ambush_2', 'game/input/ambush_4', 'game/input/ambush_5', 'game/input/ambush_6', 'game/input/ambush_7']
    # INPUT_DIRS = ['../test_mpi/alley_2', '../test_mpi/ambush_5', '../test_mpi/bandage_2', '../test_mpi/market_6', '../test_mpi/temple_2']
    INPUT_DIRS = ['../../unity_games_test/game_1_demo_scene_1', '../../unity_games_test/game_1_demo_scene_2', '../../unity_games_test/game_1_demo_scene_3',
                  '../../unity_games_test/game_2_fontaine_scene_1', '../../unity_games_test/game_2_fontaine_scene_2', '../../unity_games_test/game_2_fontaine_scene_3',
                  '../../unity_games_test/game_3_dead_scene_1', '../../unity_games_test/game_3_dead_scene_2', '../../unity_games_test/game_3_dead_scene_3',
                  '../../unity_games_test/game_4_seed_scene_1', '../../unity_games_test/game_4_seed_scene_2', '../../unity_games_test/game_4_seed_scene_3']

    # INPUT_DIRS = ['../../unity_games_test/game_1_demo_scene_1', '../../unity_games_test/game_1_demo_scene_2', '../../unity_games_test/game_1_demo_scene_3',
    #               '../../unity_games_test/game_2_fontaine_scene_1', '../../unity_games_test/game_2_fontaine_scene_2',
    #               '../../unity_games_test/game_4_seed_scene_1']

    INPUT_DIRS = ['../../art_fid_frames_content']
    # HPC
    # INPUT_DIRS = ['../../../../fastdata/acp20ei/sintel/input/ambush_2', '../../../../fastdata/acp20ei/sintel/input/ambush_4', '../../../../fastdata/acp20ei/sintel/input/ambush_5', '../../../../fastdata/acp20ei/sintel/input/ambush_6', '../../../../fastdata/acp20ei/sintel/input/ambush_7'] 

    sum_content = 0.
    sum_style = 0.
    #############################################################################
    for input_dir in INPUT_DIRS:

        # retrieve the original frames
        # original_frames_path = 'game/input/ambush_2_frames/*.png'
        original_frames_path = input_dir + '/*.jpg'
        original_frames = []
        for img in sorted(glob.glob(original_frames_path)):
            image = Image.open(img).convert('RGB')
            #if isinstance(image, Image.Image):
            
            image = transforms.Resize((args.image_size,args.image_size))(image)
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(0).to(device)
            original_frames.append(image)
        print('original frames: ', len(original_frames))



        # retrive content images
        # style = 'composition_vii'
        scene_name = input_dir.replace('../../unity_games_test/','')
        # dir_name = input_dir.replace('../test_mpi/', 'game/output/').replace(scene_name,'')
        # uncomment 2 lines below for HPC
        # scene_name = input_dir.replace('../../../../fastdata/acp20ei/sintel/input/','')
        # dir_name = '../sintel_output/'
        
        test_frames_path = args.frames
        all_scenes = get_subdirectories(args.frames)
        for scene in all_scenes:
            if scene_name in str(scene):
                test_frames_path += scene

        test_frames_path = args.frames
        #test_frames_path = dir_name + args.model.replace('saved_models/','').replace('.pth', '_test_' + scene_name)
        # test_frames_path = 'game/fast-multi-style-results' + style + scene_name + '_frames'
        #test_frames_path = input_dir # args.frames
        print(test_frames_path)
        content_images = []
        for img in sorted(glob.glob(test_frames_path + '/*.*')):         
            image = Image.open(img).convert('RGB')
            #if isinstance(image, Image.Image):
            
            image = transforms.Resize((args.image_size,args.image_size))(image)
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(0).to(device)
            content_images.append(image)
        print('content images: ', len(content_images))


        content_error = 0.
        style_error = 0.
        for itr, (org, stylized) in enumerate(zip(original_frames, content_images)):
            features_org = vgg(org)
            features_stylized = vgg(stylized)
            content_error += mse_loss(features_stylized.relu2_2, features_org.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_stylized, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s)
            style_error += style_loss

        print('Average content error: ', content_error.item()/len(original_frames))
        print('Average style error: ', style_error.item()/len(original_frames))
        sum_content += content_error/len(original_frames)
        sum_style += style_error/len(original_frames)
        
    
    print('Average content error over all directories: ', round(sum_content.item()/len(INPUT_DIRS),4))
    print('Average style error over all directories: ', round(sum_style.item()/len(INPUT_DIRS),8))
        

    
if __name__ == "__main__":
    main()