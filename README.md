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

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. I used COCO 2014 Training images dataset [80K/13GB] [(download)](https://cocodataset.org/#download).
* `--style-image`: path to style-image.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.
* `--content-weight`: weight for content-loss, default is 1e5.
* `--style-weight`: weight for style-loss, default is 1e10.
* `--depth-loss`: set it to 1 to train with depth loss, 0 train without depth loss, default is 1.
* `--depth-weight`: weight for depth-loss, default is 1e5


## Injecting NST in a Computer Game
### Setup
* Unity 2021.3.25f1
* High Definition RP 12.1.11
* Barracuda 3.0.0

![Unity Setup](images/unity_setup_screenshot.png)

