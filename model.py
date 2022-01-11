import librosa
import numpy as np
import torch
from torch import nn
from resnet_dropout import *
from stft import STFT

class Classifier(nn.Module):
    def __init__(self, num_classes=10, resnet=resnet18, nfft=1024, hop_length=512,
                 window="hanning", sample_rate=22050, num_mels=128,
                 log_base=10, parts=128, length=1024):
        super(Classifier, self).__init__()
        self.length = length
        self.split_parts = parts
        self.stft = STFT(nfft=nfft, hop_length=hop_length, window=window,
                         sample_rate=sample_rate, num_mels=num_mels, log_base=log_base)
        self.resnet = resnet(num_classes=num_classes)

    def load_resnet_weights(self, path):
        self.resnet.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"Loaded resnet weights from: {path}")

    def forward(self, x):
        x = self.stft(x)
        x = self.resize_array(x)
        # x = self.split_into_batch(x)
        x = self.monochrome2RGB(x)
        return self.resnet(x)

    def resize_array(self, tensor):
        return tensor[..., :self.length]

    def split(self, tensor):
        return torch.split(tensor, self.parts, dim=-1)

    def split_into_batch(self, tensor):
        return torch.concat(self.split(tensor), dim=0)

    @staticmethod
    def monochrome2RGB(tensor):
        if len(tensor.shape) == 4 and tensor.shape[1] == 1:
            return tensor.repeat(1, 3, 1, 1)
        return tensor.unsqueeze(1).repeat(1, 3, 1, 1)


if __name__ == "__main__":
    model = Classifier()
    path = r"C:\Users\elata\code\MusicGenreClassifier\datasets\genres\blues\blues.00029.wav"
    tensor, sr = librosa.load(path, 22050)
    # tensor = tensor - np.mean(tensor)
    tensor = torch.from_numpy(tensor[np.newaxis, :])
    out = model(tensor)
    pass
