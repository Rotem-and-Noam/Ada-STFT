import os

import librosa
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.signal
import matplotlib.pyplot as plt


class STFT(nn.Module):
    def __init__(self, nfft=1024, hop_length=512, window="hanning", sample_rate=22050, num_mels=128, log_base=10,
                 learn_window=False, learn_kernels=False):
        super(STFT, self).__init__()
        assert nfft % 2 == 0

        self.hop_length = hop_length
        self.n_freq = nfft//2 + 1
        self.nfft = nfft
        self.num_mels = num_mels
        self.log_base = log_base
        self.sample_rate = sample_rate
        self.initial_window, self.initial_kernel = self._init_kernels(nfft, window)
        self.win_cof = nn.Parameter(torch.clone(self.initial_window), requires_grad=learn_window)
        self.kernels = nn.Parameter(torch.clone(self.initial_kernel), requires_grad=learn_kernels)
        self.real_kernels, self.imag_kernels = None, None

    def forward(self, sample):

        if self.training and (self.win_cof.requires_grad or self.kernels.requires_grad):
            # recalculate kernels for STFT calculation as window or DFT kernels might have changed
            self.real_kernels, self.imag_kernels = None, None
            real_kernels, imag_kernels = self._get_stft_kernels()
        else:
            # remember kernels for STFT calculation as window or DFT kernels don't change
            if self.real_kernels is None or self.imag_kernels is None:
                self.real_kernels, self.imag_kernels = self._get_stft_kernels()
            real_kernels, imag_kernels = self.real_kernels, self.imag_kernels

        sample = sample.reshape(sample.shape[0], 1, 1, -1)

        magn = F.conv2d(sample, real_kernels, stride=self.hop_length)
        magn = magn.permute(0, 2, 1, 3)
        magn = torch.abs(magn).squeeze(dim=0)

        # apply transform to Mel scale or log if set
        if self.num_mels is not None:
            magn = self.apply_mel(magn, num_mels=self.num_mels)
        if self.log_base is not None:
            magn = self.apply_log(magn, self.log_base)
        return magn

    def _init_kernels(self, nfft, window):
        nfft = int(nfft)
        assert nfft % 2 == 0

        def kernel_fn(freq, time):
            return np.exp(-1j * (2 * np.pi * time * freq) / float(nfft))

        kernels = np.fromfunction(kernel_fn, (nfft//2+1, nfft), dtype=np.float64)

        win_cof = scipy.signal.get_window(window, nfft)[np.newaxis, :]

        initial_window = torch.from_numpy(win_cof)
        initial_kernel = torch.from_numpy(kernels)

        return initial_window, initial_kernel


    def _get_stft_kernels(self):
        # real and imaginary kernel calculation from window and DFT kernel calculation
        kernels = self.kernels[:, np.newaxis, np.newaxis, :] * self.win_cof

        real_kernels = torch.real(kernels).float()
        imag_kernels = torch.imag(kernels).float()
        # imaginary kernels are not used as we are not interested in the phase
        # this is kept for future use
        return real_kernels, imag_kernels

    def apply_mel(self, stft, num_mels=128):
        mel_basis = librosa.filters.mel(self.sample_rate, n_fft=self.nfft, n_mels=num_mels)
        return torch.from_numpy(mel_basis).unsqueeze(dim=0).to(stft.device) @ stft

    def apply_log(self, stft, log_base=10):
        return torch.log10(1 + log_base * stft)

    def learn_window(self):
        self.win_cof.requires_grad = True

    def learn_kernels(self):
        self.kernels.requires_grad = True

    def print_learnable_params(self):
        if self.win_cof.requires_grad:
            print(f"Learning window")
        if self.kernels.requires_grad:
            print(f"Learning kernels")

    def calc_window_change(self):
        return torch.nn.functional.mse_loss(self.win_cof.detach(), self.initial_window.to(self.win_cof.device))

    def calc_kernels_change(self):
        return torch.norm(self.kernels.detach() - self.initial_kernel.to(self.win_cof.device))


if __name__ == "__main__":
    # print audio spectrogram for visualization
    path = r"/home/tiras/adastft/dataset/genres/blues/blues.00029.wav"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    array, sr = librosa.load(path, 22050)

    tensor = torch.from_numpy(array[np.newaxis, :]).to(device)
    slice = tensor.shape[1] // 6
    tensor_ = tensor[:, :slice]
    model = STFT(window="hanning").to(device)
    stft = model.forward(tensor_).detach().to("cpu")

    plt.imshow(stft.squeeze().detach().numpy())
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.axis("off")
    plt.show()

    plt.plot(tensor_.detach().to("cpu").numpy()[0])
    plt.xlabel("time")
    plt.axis("off")
    plt.show()
