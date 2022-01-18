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


# Ada-STFT
Expanding on existing application of image processing networks to audio using STFT, we propose an adaptive STFT layer that learns the best DFT kernel and window for the application. 

The task of audio-processing using neural networks has proven to be a difficult task, even for the state of the art 1-Dimension processing network.
The use of STFT to transform an audio-processing challenge into an image-processing challenge enables the use of better and stronger image-processing networks.
An example of such uses can be found in following paper https://arxiv.org/abs/1706.07156.
Because STFT is in essence a feature extractor, base on applying 1-Dimension convolutions, we propose a method to simplify the translation of 1-D sequences into 2-D images.
We will also improve the vanilla STFT by learning task-specific STFT window coefficients and DFT kernal coefficients, using pytorch's build in capabilities.

In this project, we implemented a toy example of an audio-processing problem - music genre classification - to show the advantages of Ada-STFT.
We have tried to classify the genre of an audio part from the GTZAN dataset (https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/code).
The music classification task is based on a project done in the technion in 2021, and can be found here https://github.com/omercohen7640/MusicGenreClassifier.

# Results

# Usage
### STFT Layer Usage Example
```python
import torch
from torch import nn
from resnet_dropout import *
from stft import STFT

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

### Training Music Genre Classifier
To train our classifier network, run `train_env.py`.
```cmd
python ./train_env.py --test_name run_basic
```
Test parameters are automatically loaded from the options.json in the project directory.
Changes to the parameters can be applied by changing the `options.json` or running with command line arguments, for example:
```cmd
python ./train_env.py --test_name run_learn_window --learn_window 1
```

### Testing Music Genre Classifier
Run the `test.py` with the `test_name` argument set to the name of the model being tested.
Setting the `test_name` argument can be done through `options.json` or command line:
```cmd
python ./test.py --test_name run_learn_window
```

# STFT Layer Parameters
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


## Prerequisites
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
* Animation by <a href="https://medium.com/@gumgumadvertisingblog">GumGum</a>.

## Agenda
- [Ada-STFT](#Ada-STFT)
- [Results](#Results)
- [Usage](#Usage)
  - [STFT Layer](#STFT Layer)
  - [Training](#Training Music Genre Classifier)
  - [Testing](#Testing Music Genre Classifier)
- [STFT Layer Parameters](#STFT Layer Parameters)
- [Prerequisites](#Prerequisites)

