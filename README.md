# Neural Style Transfer for Computer Games

The repository includes code for training a depth-aware NST network and for injecting such a network as part of a game's 3D rendering pipeline. This is achieved using a Custom Pass applied in Unity's HDRP. 

## Depth-aware Stylisation Network
### Setup
* Python 3.8
* [PyTorch](http://pytorch.org/) 2.0.1+cu117
* [Kornia](https://kornia.readthedocs.io/) 0.6.12
* [ONNX](https://onnx.ai/) 1.14.0
* [ONNX Runtime](https://onnxruntime.ai/) 1.15.1

### Datasets
* MS COCO 2014 Training images dataset [[80K/13GB](https://cocodataset.org/#download)]
* MPI Sintel Training Images [[1.7GB](http://sintel.is.tue.mpg.de/downloads)]

### Usage
Stylize image
```
python depth-aware-nst/fast_neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `starry_night.pth`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Train model
```bash
python depth-aware-nst/fast_neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below:
* `--style-image`: path to style-image.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.
* `--content-weight`: weight for content-loss, default is 1e5.
* `--style-weight`: weight for style-loss, default is 1e10.
* `--depth-loss`: set it to 1 to train with depth loss, 0 train without depth loss, default is 1.
* `--depth-weight`: weight for depth-loss, default is 1e5
* `--feats`: set it to 1 to train with depth-of-gaussian loss


## Injecting NST in a Computer Game
### Setup
* Unity 2021.3.25f1
* [High Definition RP 12.1.11](https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@12.1/manual/index.html)
* [Barracuda 3.0.0](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/index.html)

![Unity Setup](images/unity_setup_screenshot.png)

* Create an empty GameObject in the scene
* Add a Custom Pass Volume
* Add the `CopyPassStylization` pass
* Fill in the missing assets (drag & drop)
    * Model Asset: The trained stylisation network exported to .onnx format
    * Style Transfer Shader: `StyleTransferShader.compute`
    * Output Render Texture: `Stylized Copy Pass.renderTexture`
    * Fullscreen Material: `FullScreen_Fullscreen_NST.mat`
    * Gbuffer Shader: `GbufferShader.compute` (This should be located at the `Resources` directory in the project's Assets folder)
    
When setting this Gameobject active in the scene, the stylization effect takes place.