# Adaptive STFT: Classify Music Geners with a learnable spectogram layer
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

# Usage
# STFT layer
```
from stft import STFT
class Classifier(nn.Module):
    def __init__(self, num_classes=10, resnet=resnet18, nfft=1024, hop_length=512,
                 window="hanning", sample_rate=22050, num_mels=128,
                 log_base=10, parts=12, length=1024):
        super(Classifier, self).__init__()
        self.stft = STFT(nfft=nfft, hop_length=hop_length, window=window,
                         sample_rate=sample_rate, num_mels=num_mels, log_base=log_base)
        self.resnet = resnet(num_classes=num_classes)

    def forward(self, x):
        x = self.stft(x)
        x = self.resize_array(x)
        # x = self.split_into_batch(x)
        x = self.monochrome2RGB(x)
        return self.resnet(x)
```

# Testing Music Genre Classifier

# Training Music Genre Classifier
```python
from PencilDrawingBySketchAndTone import *
import matplotlib.pyplot as plt
ex_img = io.imread('./inputs/11--128.jpg')
pencil_tex = './pencils/pencil1.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=0, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                       stroke_darkness= 2,tone_darkness=1.5)
plt.rcParams['figure.figsize'] = [16,10]
plt.imshow(ex_im_pen)
plt.axis("off")
```
# Parameters
* kernel_size = size of the line segement kernel (usually 1/30 of the height/width of the original image)
* stroke_width = thickness of the strokes in the Stroke Map (0, 1, 2)
* num_of_directions = stroke directions in the Stroke Map (used for the kernels)
* smooth_kernel = how the image is smoothed (Gaussian Kernel - "gauss", Median Filter - "median")
* gradient_method = how the gradients for the Stroke Map are calculated (0 - forward gradient, 1 - Sobel)
* rgb = True if the original image has 3 channels, False if grayscale
* w_group = 3 possible weight groups (0, 1, 2) for the histogram distribution, according to the paper (brighter to darker)
* pencil_texture_path = path to the Pencil Texture Map to use (4 options in "./pencils", you can add your own)
* stroke_darkness = 1 is the same, up is darker.
* tone_darkness = as above

Credits:
* Animation by <a href="https://medium.com/@gumgumadvertisingblog">GumGum</a>.
* Reference work by <a href="https://github.com/Dohppak/Music_Genre_Classification_Pytorch">Dohppak</a>.

## Agenda
- [Agenda](#agenda)
- [MusicGenreClassifier](#MusicGenreClassifier)
- [Dataset](#Dataset)
- [Data augmentation](#Data-augmentation)
- [1D-Classifier](#1D-Classifier)
- [2D-Classifier](#2D-Classifier)
  * [Feature extraction](#Feature-extraction)
  * [Ensaemble](#Ensemble)
