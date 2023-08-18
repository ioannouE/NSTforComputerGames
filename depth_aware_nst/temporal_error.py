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
    parser.add_argument("--image-size", type=int, default=256,
                                  help="the image size")
    
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

     # set up midas
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: ", torch.cuda.get_device_name(0))
    print("Running on ", device)

    mse_loss = torch.nn.MSELoss()

    # INPUT_DIRS = ['game/input/ambush_2', 'game/input/ambush_4', 'game/input/ambush_5', 'game/input/ambush_6', 'game/input/ambush_7']
    # INPUT_DIRS = ['../test_mpi/alley_2', '../test_mpi/ambush_5', '../test_mpi/bandage_2', '../test_mpi/market_6', '../test_mpi/temple_2']
    INPUT_DIRS = ['../../unity_games_test/game_1_demo_scene_1', '../../unity_games_test/game_1_demo_scene_2', '../../unity_games_test/game_1_demo_scene_3',
                  '../../unity_games_test/game_2_fontaine_scene_1', '../../unity_games_test/game_2_fontaine_scene_2', '../../unity_games_test/game_2_fontaine_scene_3',
                  '../../unity_games_test/game_3_dead_scene_1', '../../unity_games_test/game_3_dead_scene_2', '../../unity_games_test/game_3_dead_scene_3',
                  '../../unity_games_test/game_4_seed_scene_1', '../../unity_games_test/game_4_seed_scene_2', '../../unity_games_test/game_4_seed_scene_3']

    INPUT_DIRS = ['../../unity_games_test/game_1_demo_scene_3']
    # HPC
    # INPUT_DIRS = ['../../../../fastdata/acp20ei/sintel/input/ambush_2', '../../../../fastdata/acp20ei/sintel/input/ambush_4', '../../../../fastdata/acp20ei/sintel/input/ambush_5', '../../../../fastdata/acp20ei/sintel/input/ambush_6', '../../../../fastdata/acp20ei/sintel/input/ambush_7'] 

    sum = 0.
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

        # retrieve the original flows
        # original_flows_path = 'game/input/ambush_2_flow/'
        original_flows_path = input_dir + '_flow/'
        original_flows = []
        for fl in sorted(os.listdir(original_flows_path)):  
            if fl.endswith('.npy'):       
                # image = cv2.imread(img)
                # image = cv2.copyMakeBorder(image,10,10,10,20,cv2.BORDER_CONSTANT,value=[255,255,255])
                # flow = readFlow(os.path.join(original_flows_path, fl))
                flow = np.load(os.path.join(original_flows_path, fl), allow_pickle=True)

                originalflow = torch.from_numpy(flow)                
                flow = torch.from_numpy(transform.resize(flow, (args.image_size,args.image_size))).to(device).permute(2,0,1).float()
                #cflow = torch.from_numpy(flow).to(device)#.permute(2,0,1).float()
                flow[0, :, :] *= float(flow.shape[1])/originalflow.shape[1]
                flow[1, :, :] *= float(flow.shape[2])/originalflow.shape[2]
                flow = flow.unsqueeze(0)
                # print(flow.size())
                original_flows.append(flow)
        #############################################################################
        print('flows: ', len(original_flows))


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
                test_frames_path += '/' + scene

        test_frames_path = args.frames
        #test_frames_path = dir_name + args.model.replace('saved_models/','').replace('.pth', '_test_' + scene_name)
        # test_frames_path = 'game/fast-multi-style-results' + style + scene_name + '_frames'
        #test_frames_path = input_dir # args.frames
        print(test_frames_path)
        content_images = []
        for img in sorted(glob.glob(test_frames_path + '/*.jpg')):         
            image = Image.open(img).convert('RGB')
            #if isinstance(image, Image.Image):
            
            image = transforms.Resize((args.image_size,args.image_size))(image)
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(0).to(device)
            content_images.append(image)
        print('content images: ', len(content_images))


        total_mse_loss = 0.
        warped_content = content_images[1]
        for itr, (img, flow) in enumerate(zip(content_images, original_flows)):
            # print(itr)
            if(itr==0):
                continue
            else:
                total_mse_loss += mse_loss(warped_content, img)
            # warped_org = warp(org, flow)
            warped_content = warp(img,flow,device)

            # plt.imshow(warped_org.squeeze().permute(1,2,0).cpu().numpy())
            # plt.show()
            
            # total_mse_loss += mse_loss(warped_content, warped_content)
            
            #print(img.squeeze().permute(1,2,0).size())
            #print(warped)
            # io.imsave(args.opticflow + "warped.png", warped.squeeze().permute(1,2,0).cpu().numpy())
        print('Average warped error: ' + test_frames_path + '  :', round(total_mse_loss.item()/len(original_flows), 4))
        sum += total_mse_loss/len(original_flows)
    
    print('Average warped error over all directories: ', sum/len(INPUT_DIRS))
    

# def warp(img, flow):
#     flow = flow.cpu()
#     img = img.cpu()
#     h, w = flow.shape[:2]
#     flow = -flow
#     flow[:,:,0] += np.arange(w)
#     flow[:,:,1] += np.arange(h)[:,np.newaxis]
#     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
#     return res

def warp(x, flo, device):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = torch.nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device) #.cuda()
    mask = torch.nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask


# TAG_CHAR = np.array([202021.25], np.float32)

# def readFlow(fn):
#     """ Read .flo file in Middlebury format"""
#     # Code adapted from:
#     # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

#     # WARNING: this will work on little-endian architectures (eg Intel x86) only!
#     # print 'fn = %s'%(fn)
#     with open(fn, 'rb') as f:
#         magic = np.fromfile(f, np.float32, count=1)
#         if 202021.25 != magic:
#             print('Magic number incorrect. Invalid .flo file')
#             return None
#         else:
#             w = np.fromfile(f, np.int32, count=1)
#             h = np.fromfile(f, np.int32, count=1)
#             # print 'Reading %d x %d flo file\n' % (w, h)
#             data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
#             # Reshape data into 3D array (columns, rows, bands)
#             # The reshape here is for visualization, the original code is (w,h,2)
#             return np.resize(data, (int(h), int(w), 2))


def readFlow(name):
    # if name.endswith('.pfm') or name.endswith('.PFM'):
    #     return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def read(file):
    # if file.endswith('.flo'): return readFlow(file)
    if file.endswith('.npy'): return readFlow(file)
    else: raise Exception('don\'t know how to read %s' % file)



if __name__ == "__main__":
    main()