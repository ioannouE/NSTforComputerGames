import argparse
import os
import sys
import time
import re

import numpy as np
import torch
torch.cuda.empty_cache()

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision
import torch.onnx

import utils
from transformer_net import TransformerNet
from transformer_net_light import TransformerNetLight
from vgg import Vgg16


# for depth use MiDaS (https://pytorch.org/hub/intelisl_midas_v2/)
import torchvision.models as models

from decimal import Decimal

import matplotlib.pyplot as plt
from pylab import *
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from tqdm import trange

from aim import Run 
from aim import Image as aimImage

from dataset import MonkaaDataset, FlyingThings3DDataset
import custom_transforms

from torchvision.utils import make_grid


def stylize_image_aim(image, model):
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
        image = transforms.Lambda(lambda x: x.mul(255))(image)
    image = image.cuda().unsqueeze_(0)
    # image = preprocess_for_reconet(image)
    styled_image = model(image)[0].squeeze()
    # styled_image = postprocess_reconet(styled_image)
    return styled_image

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):

    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: ", torch.cuda.get_device_name(0))
    # calculate the save file name
    save_model_filename = str(time.ctime()).replace(':','_').replace(' ', '_') + "_" + str(
        args.style_image).replace('.jpg', '').replace('images/styles/', '') + "_" + str(
            args.depth_loss) + "_" + str('%.E' % Decimal (args.content_weight)).replace('+','') + "_" + str(
            '%.E' % Decimal (args.style_weight)).replace('+','') + "_" + str(
                '%.E' % Decimal (args.depth_weight)).replace('+', '') + ".pth" # changed extension to .pth
    print('Save model filename: ', save_model_filename)

    if (args.depth_loss):
        print("Training with depth loss (MiDaS), depth weight = ", args.depth_weight)
    else:
        print("Training without depth loss")

    # writer = SummaryWriter()
    if (args.aim):
        run = Run(experiment="Depth + Semantic")

        # Log run parameters
        run["hparams"] = {   
            "epochs": args.epochs,
            "content_loss": args.content_weight,
            "style_loss": args.style_weight,
            "depth_loss": args.depth_loss,
            "depth_weight": args.depth_weight,
            "seg_loss": args.sem_loss,
            "seg-weight": args.sem_weight

        }

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform_coco = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    transform_sintel = transforms.Compose([
        # transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255))
    ])

    # transform = transforms.Compose([
    #     custom_transforms.Resize(640, 360),
    #     custom_transforms.RandomHorizontalFlip(),
    #     custom_transforms.ToTensor()
    # ])

    coco = datasets.ImageFolder('../train', transform_coco)
    sintel = datasets.ImageFolder(args.dataset, transform_sintel)   
    wikiart = datasets.ImageFolder('../wikiart', transform_coco)
    train_dataset = coco + wikiart
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # monkaa = MonkaaDataset(os.path.join(args.dataset, "monkaa"), transform)
    # flyingthings3d = FlyingThings3DDataset(os.path.join(args.dataset, "flyingthings3d"), transform)
    # dataset = monkaa + flyingthings3d
    # print(dataset)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
    #                                             num_workers=3,
    #                                             pin_memory=True,
    #                                             drop_last=True)
    
    transformer = TransformerNetLight() # .to(device)
    if (args.pretrained_model):
        state_dict = torch.load(args.pretrained_model)
        transformer.load_state_dict(state_dict)
    transformer.to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)


    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    #######################################################################################
    # depth estimation network (MiDaS)
    if (args.depth_loss):
        model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # ##model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # ##model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            midas_transform = midas_transforms.dpt_transform
        else:
            midas_transform = midas_transforms.small_transform
        
        for param in midas.parameters():
            param.requires_grad = False
        midas = midas.to(device)         
        midas.eval()

        ###########################
    if (args.sem_loss):
        # semantic segmentation
        deeplab = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        deeplab.to(device)
        deeplab.eval()


       
    log_count = 0
    global_step = 0
    for e in range(args.epochs):
        
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.

        if (args.depth_loss):
            agg_depth_loss = 0.
        
        if (args.sem_loss):
            agg_sem_loss = 0.

        count = 0

        # for batch_id, (x, _) in enumerate(train_loader):
        # added code for displaying progress bar 
        with tqdm(train_loader, unit="batch") as tepoch:

            tepoch.set_description(f"Epoch {e + 1}")
            # for sample in traindata:
            for (x, _) in tepoch:
                # sample = {name: tensor.cuda() for name, tensor in sample.items()}

                # x_numpy = x[0].permute(1,2,0).cpu().numpy()
                # plt.imshow(x_numpy)
                # plt.show()
                    
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()
                

                x = x.to(device)
                # x_clean = sample["frame_clean"].to(device)

                y = transformer(x)
                # y_clean = transformer(x_clean)

                img_grid = y
                
                features_y = vgg(y)
                features_x = vgg(x)

                content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)
                
                style_loss = 0.
                for ft_y, gm_s in zip(features_y, gram_style):
                    gm_y = utils.gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s[:args.batch_size, :, :])
                style_loss *= args.style_weight 


                # frame_loss = 1e2 * mse_loss(y_final, y_clean)
                total_loss = content_loss + style_loss # + frame_loss

                # depth loss
                if (args.depth_loss): 
                    x_midas = midas(x)
                    y_midas = midas(y)
                    depth_loss = args.depth_weight * mse_loss(y_midas, x_midas)
                    total_loss += depth_loss 
                # else:
                #     total_loss = content_loss + style_loss

                
                if (args.sem_loss):
                    ## semantic
                    x_sem = deeplab(x)['out'][0]
                    y_sem = deeplab(y)['out'][0]

                    semantic_loss = args.sem_weight * mse_loss(y_sem, x_sem)
                    total_loss += semantic_loss 
                    
                    
                    # output_predictions = x_sem.argmax(0)

                    # # create a color pallette, selecting a color for each class
                    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                    # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
                    # colors = (colors % 255).numpy().astype("uint8")

                    # # plot the semantic segmentation predictions of 21 classes in each color
                    # r = Image.fromarray(output_predictions.byte().cpu().numpy())#.resize(256)
                    # r.putpalette(colors)

                    # plt.imshow(r)
                    # plt.show()


                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()
                if (args.depth_loss):
                    agg_depth_loss += depth_loss.item()
                
                if (args.sem_loss):
                    agg_sem_loss += semantic_loss.item()
                
                if global_step % args.log_interval == 0:
                    if ((not args.depth_loss) and (not args.sem_loss)):
                        mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                            time.ctime(), e + 1, count, len(train_dataset),
                                        agg_content_loss / (global_step + 1),
                                        agg_style_loss / (global_step + 1),
                                        (agg_content_loss + agg_style_loss) / (global_step + 1)
                        )
                        # writer.add_scalar("content_loss", agg_content_loss / (batch_id + 1), log_count)
                        # writer.add_scalar("style_loss", agg_style_loss / (batch_id + 1), log_count)
                        # writer.add_images('stylised_result', img_grid, log_count)

                        if (args.aim):
                            run.track(agg_content_loss / (global_step + 1), name='content loss', epoch=e, context={'subset': 'val'})
                            run.track(agg_style_loss / (global_step + 1), name='style loss', epoch=e, context={'subset': 'val'})
                            for i in range(0, len(train_dataset), len(train_dataset) // 4):
                                sample = train_dataset[i]
                                styled_train_image_1 = stylize_image_aim(x[0], transformer)
                                styled_train_image_2 = stylize_image_aim(x[1], transformer)
                                # styled_train_image_1 = stylize_image_depth(sample["frame"], sample["disparity_frame"], model, depthEncoder)
                                # styled_train_image_2 = stylize_image_depth(sample["previous_frame"], sample["disparity_previous_frame"], model, depthEncoder)
                                # styled_train_image_1 = stylize_image_flow(sample["frame"], sample["optical_flow"], model, depthEncoder)
                                # styled_train_image_2 = stylize_image_flow(sample["previous_frame"], sample["reverse_optical_flow"], model, depthEncoder)
                                grid = torchvision.utils.make_grid([styled_train_image_1, styled_train_image_2])
                            # print(img_grid.size())
                            run.track(aimImage(grid), name='train image', context={ "subset": "train" })


                            # run.track(aimImage(img_grid), name='train image', context={ "subset": "train" })

                    elif (args.depth_loss and not args.sem_loss):
                        mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tdepth: {:.6f}\ttotal: {:.6f}".format(
                            time.ctime(), e + 1, count, len(train_dataset),
                                        agg_content_loss / (global_step + 1),
                                        agg_style_loss / (global_step + 1),
                                        agg_depth_loss / (global_step + 1),
                                        (agg_content_loss + agg_style_loss + agg_depth_loss) / (global_step + 1)
                        )
                        # writer.add_scalar("content_loss", agg_content_loss / (batch_id + 1), log_count)
                        # writer.add_scalar("style_loss", agg_style_loss / (batch_id + 1), log_count)
                        # writer.add_scalar("depth_loss", agg_depth_loss / (batch_id + 1), log_count)
                        # writer.add_images('stylised_result', img_grid, log_count)  

                        if (args.aim):  
                            run.track(agg_content_loss / (global_step + 1), name='content loss', epoch=e, context={'subset': 'val'})
                            run.track(agg_style_loss / (global_step + 1), name='style loss', epoch=e, context={'subset': 'val'})
                            run.track(agg_depth_loss / (global_step + 1), name='depth loss', epoch=e, context={'subset': 'val'})

                            run.track(aimImage(img_grid), name='train image', context={ "subset": "train" })

                    elif (args.sem_loss and not args.depth_loss):
                        mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tsemantic: {:.6f}\ttotal: {:.6f}".format(
                            time.ctime(), e + 1, count, len(train_dataset),
                                        agg_content_loss / (global_step + 1),
                                        agg_style_loss / (global_step + 1),
                                        agg_sem_loss / (global_step + 1),
                                        (agg_content_loss + agg_style_loss + agg_sem_loss) / (global_step + 1)
                        )
                        # writer.add_scalar("content_loss", agg_content_loss / (batch_id + 1), log_count)
                        # writer.add_scalar("style_loss", agg_style_loss / (batch_id + 1), log_count)
                        # writer.add_scalar("sem_loss", agg_sem_loss / (batch_id + 1), log_count)
                        # writer.add_images('stylised_result', img_grid, log_count)    
                        if (args.aim):
                            run.track(agg_content_loss / (global_step + 1), name='content loss', epoch=e, context={'subset': 'val'})
                            run.track(agg_style_loss / (global_step + 1), name='style loss', epoch=e, context={'subset': 'val'})
                            run.track(agg_sem_loss / (global_step + 1), name='semantic loss', epoch=e, context={'subset': 'val'})
                            for i in range(0, len(train_dataset), len(train_dataset) // 4):
                                sample = train_dataset[i]
                                styled_train_image_1 = stylize_image_aim(sample["frame_final"], transformer)
                                styled_train_image_2 = stylize_image_aim(sample["frame_clean"], transformer)
                                # styled_train_image_1 = stylize_image_depth(sample["frame"], sample["disparity_frame"], model, depthEncoder)
                                # styled_train_image_2 = stylize_image_depth(sample["previous_frame"], sample["disparity_previous_frame"], model, depthEncoder)
                                # styled_train_image_1 = stylize_image_flow(sample["frame"], sample["optical_flow"], model, depthEncoder)
                                # styled_train_image_2 = stylize_image_flow(sample["previous_frame"], sample["reverse_optical_flow"], model, depthEncoder)
                                grid = torchvision.utils.make_grid([styled_train_image_1, styled_train_image_2])
                            # print(img_grid.size())
                            run.track(aimImage(grid), name='train image', context={ "subset": "train" })


                            # run.track(aimImage(img_grid), name='train image', context={ "subset": "train" })

                    else: 
                        mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tdepth: {:.6f}\tsemantic: {:.6f}\ttotal: {:.6f}".format(
                            time.ctime(), e + 1, count, len(train_dataset),
                                        agg_content_loss / (global_step + 1),
                                        agg_style_loss / (global_step + 1),
                                        agg_depth_loss / (global_step + 1),
                                        agg_sem_loss / (global_step + 1),
                                        (agg_content_loss + agg_style_loss + agg_depth_loss + agg_sem_loss) / (global_step + 1)
                        )
                        print(mesg)
                        # writer.add_scalar("content_loss", agg_content_loss / (batch_id + 1), log_count)
                        # writer.add_scalar("style_loss", agg_style_loss / (batch_id + 1), log_count)
                        # writer.add_scalar("depth_loss", agg_depth_loss / (batch_id + 1), log_count)
                        # writer.add_scalar("sem_loss", agg_sem_loss / (batch_id + 1), log_count)
                        # writer.add_images('stylised_result', img_grid, log_count)    
                        if (args.aim):
                            run.track(agg_content_loss / (global_step + 1), name='content loss', epoch=e, context={'subset': 'val'})
                            run.track(agg_style_loss / (global_step + 1), name='style loss', epoch=e, context={'subset': 'val'})
                            run.track(agg_depth_loss / (global_step + 1), name='depth loss', epoch=e, context={'subset': 'val'})
                            run.track(agg_sem_loss / (global_step + 1), name='semantic loss', epoch=e, context={'subset': 'val'})

                            for i in range(0, len(train_dataset), len(train_dataset) // 4):
                                sample = train_dataset[i]
                                styled_train_image_1 = stylize_image_aim(sample["frame_final"], transformer)
                                styled_train_image_2 = stylize_image_aim(sample["frame_clean"], transformer)
                                # styled_train_image_1 = stylize_image_depth(sample["frame"], sample["disparity_frame"], model, depthEncoder)
                                # styled_train_image_2 = stylize_image_depth(sample["previous_frame"], sample["disparity_previous_frame"], model, depthEncoder)
                                # styled_train_image_1 = stylize_image_flow(sample["frame"], sample["optical_flow"], model, depthEncoder)
                                # styled_train_image_2 = stylize_image_flow(sample["previous_frame"], sample["reverse_optical_flow"], model, depthEncoder)
                                grid = torchvision.utils.make_grid([styled_train_image_1, styled_train_image_2])
                            # print(img_grid.size())
                            run.track(aimImage(grid), name='train image', context={ "subset": "train" })


                global_step += 1
                log_count += 1
                # print(mesg)

                if args.checkpoint_model_dir is not None and (global_step + 1) % args.checkpoint_interval == 0:
                    transformer.eval().cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(global_step + 1) + ".pth"
                    ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                    torch.save(transformer.state_dict(), ckpt_model_path)
                    transformer.to(device).train()
                
           

    # writer.flush()
    # writer.close()
    # save model
    transformer.eval().cpu()
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNetLight() 
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    
    # rep = {".jpg": ".png", ".png": "text"} # define desired replacements here
    output_image_name = 'images/output/' + str(
        args.model).replace('saved_models/', '').replace('.pth', '_') + str(
            args.content_image).replace('images/content/','').replace('.jpg','.png')
    # utils.save_image(args.output_image, output[0])
    if (args.output_image):
        print('Output image name: ', args.output_image)
        utils.save_image(args.output_image, output[0])
    else:
        print('Output image name: ', output_image_name)
        utils.save_image(output_image_name, output[0])


def stylize_p(content_img, model, cuda=1, content_scale=None):
    device = torch.device("cuda" if cuda else "cpu")

    content_image = utils.load_image(content_img, scale=content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)


    with torch.no_grad():
        style_model = TransformerNet() # TransformerNetDepth()
        state_dict = torch.load(model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
            # if re.search(r'in\d+\.num_batches_tracked$', k):
            #     del state_dict[k]

        style_model.load_state_dict(state_dict)
        style_model.to(device)
            
        output = style_model(content_image).cpu()
    
    output_image_name = 'images/output/' + str(
        model).replace('saved_models/', '').replace('.pth', '/') + str(
            content_img).replace('images/content/','').replace('.jpg','.png')
   
    utils.save_image(output_image_name, output[0])


def stylize_image(image, model, device):
    # content_image = utils.load_image(image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(image)
    content_image = content_image.unsqueeze(0).to(device)
    styled_image = model(image)[0].squeeze()
    # styled_image = postprocess_reconet(styled_image)
    return styled_image


def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not args.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(args.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=2,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=False, default='saved_models/',
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=100,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    train_arg_parser.add_argument("--depth-loss", type=int, default=1,
                                  help="set it to 1 to train with depth loss, 0 train without depth loss, default is 1")
    train_arg_parser.add_argument("--depth-weight", type=float, default=1e3,
                                help="weight for depth-loss, default is 1e3")
    train_arg_parser.add_argument("--sem-loss", type=int, default=1,
                                  help="set it to 1 to train with semantic loss, 0 train without depth loss, default is 1")
    train_arg_parser.add_argument("--sem-weight", type=float, default=1e3,
                                help="weight for semantic-loss, default is 1e10")
    train_arg_parser.add_argument("--pretrained-model", type=str, default=None,
                                help="path to pretrained model")
    train_arg_parser.add_argument("--aim", type=int, default=0,
                                help="set to 1 to track experiment using Aim")


    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=False,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()