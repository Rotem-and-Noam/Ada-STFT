# Adaptive STFT: Classify Music Genres with a learnable spectrogram layer
<h2 align="center">
  <br>
 Our final project for the Technion's EE Deep Learning course (046211)
  <br>
  <img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nn_gumgum.gif" height="200">
</h1>
  <p align="center">
    Noam Elata: <a href="https://www.linkedin.com/in/noamelata/">LinkdIn</a> , <a href="https://github.com/noamelata">GitHub</a>
  <br>
    Rotem Idelson: <a href="https://www.linkedin.com/in/rotem-idelson/">LinkdIn</a> , <a href="https://github.com/RotemId">GitHub</a>
  </p>

## Agenda
- [Ada-STFT](#Ada-STFT) - about our project
- [Training and Results](#training-and-results) - our network's training visualizations and results
- [Hyper-parameters](#hyper-parameters) - what are our training's hyperparameters
- [Run our model](#run-our-model) - how to run training jobs and inference with our model and how to load checkpoints
- [Ada-STFT Module](#ada-stft-module) - how to use our STFT module
- [Prerequisites](#Prerequisites) - Prerequisites of the environment

# Ada-STFT
Expanding on existing application of image processing networks to audio using STFT, we propose an adaptive STFT layer that learns the best DFT kernel coefficients and window coefficients for the application. 

The task of audio-processing using neural networks has proven to be a difficult task, even for the state of the art 1-Dimension processing network.
The use of STFT to transform an audio-processing challenge into an image-processing challenge enables the use of better and stronger image-processing networks, such as Resnet.
An example of such uses can be found in this <a href="https://arxiv.org/abs/1706.07156">paper</a>.
Because STFT is in essence a feature extractor, base on applying 1-Dimension convolutions, we propose a method to simplify the translation of 1-D sequences into 2-D images.
We will also improve the vanilla STFT by learning task-specific STFT window coefficients and DFT kernal coefficients, using pytorch's build in capabilities.

In this project, we implemented a toy example of an audio-processing problem - music genre classification - to show the advantages of Ada-STFT.
We have tried to classify the genre of an audio part from the <a href="https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/code">GTZAN dataset </a>.
The music classification task is based on a project done in the technion in 2021, and can be found <a href=https://github.com/omercohen7640/MusicGenreClassifier>here</a>.

<img src="images/model.png" height="200">

A complete and detailed report of the project can be found <a href=https://github.com/Rotem-and-Noam/Ada-STFT/blob/main/Ada_STFT.pdf>here</a>

# Training and Results

We used Optuna to pick our hyperparameters for basic run with no learnable STFT's coefficiens. Those parameters are saved in the `codes\options.json` file.
You can use our code `codes\train_optuna.py` and change it if you would like to preform your own Optuna study.

With those parameters, we conducted the folowing trials:

* basic run, with no STFT learning
* learning the STFT's window coefficients
* learning the STFT's DFT's kernel coefficients
* learning both the DFT's kernel coefficients and window coefficients
* learning 3 different STFT's: window coefficients only
* learning 3 different STFT's: DFT's kernel coefficients
* learning 3 different STFT's: both DFT's kernel coefficients and window coefficients

Here are our results:

<img src="images/train_loss.png"  height="250">
<p style="text-align: center;">Train loss progress</p>

<img src="images/val_accuracy_graph.png"  height="200">
<p style="text-align: center;">Validation accuracy progress</p>

<img src="images/val_accuracy_matrix.png"  height="300">
<p style="text-align: center;">Validation confusion matrices</p>


As we can see, out of the following 3 combinations:
1. learning the STFT window coefficients
2. learning the STFT DFT kernel coefficients
3. learning both the DFT kernel coefficients and window coefficients
It appears that learning both the DFT kernel coefficients and window coefficients together has the best performance.
Surprisingly, it seems that learning 3 different STFT modules (one for each of Resnet's input channels) does not improve the performance over learning 1 STFT module;
It performs slightly better or slightly worse, depending on the trial configuration and chance.

# Run our model

## Dataset
Our dataset is: <a href="https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/code">GTZAN dataset </a>,
Our code uses `torchaudio` dataset to load it. You can set the path to your data directory with the data_dir argument.

## Chekpoints
You should set the ckpt_dir parameter as the father checkpoints directory, and ckpt_file as the file name.
For example, if you set the following parameters as:
`ckpt_dir = "checkpoints"`, `test_name = "my_test.pt"`, `ckpt_dir = "best_ckpt.pt"`,
The full checkpoints file path that will be loaded is: `\checkpoints\my_test\best_ckpt.pt`

## Training Music Genre Classifier
To train our classifier network, run `train_env.py`.
```cmd
python ./train_env.py --test_name run_basic
```
Training job parameters are automatically loaded from the options.json in the project directory.
Changes to the parameters can be applied by changing the `codes\options.json` or running with command line arguments, for example:
```cmd
python ./train_env.py --test_name run_learn_window --learn_window 1
```

## Inference Music Genre Classifier
Run the `test.py` with the `test_name` argument set to the name of the model used for inference.
Setting the `test_name` argument can be done through `options.json` or through command line:
```cmd
python ./test.py --test_name my_test --ckpt_dir checkpoints --ckpt_dir best_ckpt.pt
```

# Hyper-parameters

|Parameter | Type | Description |
|-------|------|---------------|
|test_name| string | your trial's name|
|resume| int | 0 if we start a new training run and 1 if we resume old training|
|ckpt_interval| int | epoch interval to save a new checkpoint |
|tensorboard_dir| string | path to tensorboard log directory |
|data_dir| string | path to dataset directory |
|ckpt_dir| string | path to checkpoint directory |
|ckpt_file| string | path to ckpt file to be loaded |
|learn_window| int | 1 to learn stft window coefficients, 0 not to |
|learn_kernels| int | 1 to learn stft kernels coefficients, 0 not to |
|batch_size| int | size of batch |
|num_workers| int | data loader's parameters: number of workers to pre-fetch the data |
|epoch_num| int | number of total epoches to run |
|learning_rate| int | initial optimizer's learning rate |
|split_parts| int | how many parts to split our original audio file to. can be: 1, 3, 4, 6, 12|
|gamma| int | scheduler's gamma |
|cpu| int | 0 if we want to try and run on gpu, else if we want to run on cpu |
|augmentation | int | 0 if we don't want to use augmentation, else if we do |
|three_widows| int | 0 to use 1 STFT in classifier (greyscale), else for 3 STFT modules in classifier (RGB) |
|optimizer_class| string | optimizer type: "SGD" or "AdamW" |

## Changing hyper-parameters
Parameters are automatically loaded from the options.json in the project directory.
Changes to the parameters can be applied by changing the `options.json`.
We also  implemented argparse library, so you can load your parameters with your IDE's configure or within th command line.
Examples are shown in the Run-our-model section.

# Ada-STFT Module

## How to use our module
```python
import torch
from torch import nn
from models.resnet_dropout import *
import STFT

class Classifier(nn.Module):
    def __init__(self, resnet=resnet18, window="hanning", num_classes=10):
        super(Classifier, self).__init__()
        self.stft = STFT(window=window)
        self.resnet = resnet(num_classes=num_classes)

    def forward(self, x):
        x = self.stft(x)
        x = self.monochrome2RGB(x)
        return self.resnet(x)

    @staticmethod
    def monochrome2RGB(tensor):
        return tensor.repeat(1, 3, 1, 1)
```

## STFT Layer Parameters
|Parameter | Description |
|-------|---------------------|
|nfft| window size of STFT calculation|
|hop_length | STFT hop size, or stride of STFT calculation|
| window | type of window to initialize the STFT window to, one of the windows implemented in scipy.signal|
| sample_rate | sampling rate for audio|
| num_mels | number of mel scale frequencies to use, None for don't use mel frequencies|
| log_base | base of log to apply  to STFT, None for no log|
| learn_window | should window be learned (can be set after layer initialization)|
| learn_kernels | should DFT kernel be learned (can be set after layer initialization)|


# Prerequisites
|Library         | Version |
|--------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`scipy`| `1.7.3`|
|`tqdm`| `4.62.3`|
|`librosa`| `0.8.1`|
|`torch`| `1.10.1`|
|`torchaudio`| `0.10.1`|
|`torchaudio-augmentations`| `0.2.3 (https://github.com/Spijkervet/torchaudio-augmentations)`|
|`tensorboard`| `2.7.0`|

Credits:
* Music Genre Classifier Project for classifier network architecture https://github.com/omercohen7640/MusicGenreClassifier
* Animation by <a href="https://medium.com/@gumgumadvertisingblog">GumGum</a>.
* STFT implemenation https://github.com/diggerdu/pytorch_audio

