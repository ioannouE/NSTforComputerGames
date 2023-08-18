import argparse
import os
import sys
import time

import torch.onnx
import torchvision
import torch
import utils
from transformer_net import TransformerNet
from transformer_net_light import TransformerNetLight
import onnx
import onnxruntime
import numpy as np
import re
torch.cuda.empty_cache()

import gc
gc.collect()

import torchvision.transforms as transforms
from PIL import Image
# from onnxsim import simplify

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def check_onnx_model(model_name, device):
    content_image = 'images/content/church.jpg'
    onnx_model = onnx.load(model_name)

    # convert model
    # model_simp, check = simplify(onnx_model)
    # torch.save(model_simp, "saved_models/onnx/simple.onnx")

    # assert check, "Simplified ONNX model could not be validated"
    # print(check)

    # print(onnx_model)
    onnx.checker.check_model(onnx_model)

    # model_name = "saved_models/onnx/simple.onnx"
    content_image = utils.load_image(content_image)
    content_transform = transforms.Compose([
        transforms.Resize(size = (640,640)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    ort_session = onnxruntime.InferenceSession(model_name)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out = ort_outs[0]
    img = img_out[0].transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save("images/output/church_stylized.jpg")

def export_to_onnx(model_name, device):
    image_size=640
    model = TransformerNetLight() # Light()
    state_dict = torch.load(model_name)
    # for k in list(state_dict.keys()):
    #     if re.search(r'in\d+\.running_(mean|var)$', k):
    #         del state_dict[k]
    
    model.load_state_dict(state_dict)
    
    for m in model.modules(): 
        # print(m.__class__.__name__.lower())
        if 'instance' in m.__class__.__name__.lower(): 
            m.eval()
    model.to(device)
    model.eval()

    # torch.onnx.register_custom_op_symbolic("com::PixelShuffleCustomOp", PixelShuffleCustomOp.forward, opset_version=9)

    onnx_model_name = model_name.replace(".pth", ".onnx").replace('to_onnx', 'onnx')
    print(onnx_model_name)
    dummy_input = torch.randn(1, 3, image_size, image_size, device = device, requires_grad=True)
    torch.onnx.export(model, dummy_input, onnx_model_name, 
                      export_params=True, 
                      input_names=['input'],
                      output_names=['output'],
                      # dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}},
                      verbose=True, 
                      opset_version=9, 
                      do_constant_folding=True) # Replace operations that have all constant inputs with pre-computed nodes

    ########################################################################
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    check_onnx_model(onnx_model_name, device)
    ########################################################################    
    x = torch.randn(1, 3, image_size, image_size, device=device, requires_grad=False)

    torch_out = model(x)
    ort_session = onnxruntime.InferenceSession(onnx_model_name)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main():
    parser = argparse.ArgumentParser(description='parser for evaluating a model')
    parser.add_argument("--model", type=str, required=True,
                                 help="saved model to export to onnx. File should end in .pth")
    parser.add_argument("--cuda", type=int, required=False, default=1,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    
    
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        print("cuda is available, running on GPU")
        device = "cuda"
    else:
        print("ERROR: cuda is not available, running on CPU")
        device = "cpu"
        # sys.exit(1)

    model_name = args.model
    export_to_onnx(model_name, device)
    


if __name__ == "__main__":
    main()