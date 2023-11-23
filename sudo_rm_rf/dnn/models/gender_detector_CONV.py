import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import librosa
import numpy as np

import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os


class GenderDetectorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(0)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(579, 512),

        )

    def forward(self, x):
        x = self.flatten(self.extract_features(x))
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits

    def extract_features(self, signal):
        X = signal.numpy(force=True)
        # print(X)
        # print(f"input size is {signal.size()}")

        sample_rate = 8000
        # Sets the name to be the path to where the file is in my computer
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
        stft = np.abs(librosa.stft(X))
        # Computes a chromagram from a waveform or power spectrogram.
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        # Computes a mel-scaled spectrogram.
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        # Computes spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate, fmin=60).T, axis=0)
        # Computes the tonal centroid features (tonnetz)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                  sr=sample_rate, fmin=20).T, axis=0)

        os.chdir("/home/dsi/yechezo/sudo_rm_rf_/sudo_rm_rf/temp")
        # print(X.shape)
        # print(np.sum(X[0,:,:],axis=0).shape)
        write("temp1.wav", 8000, X[0, 0, :])
        write("temp2.wav", 8000, X[0, 1, :])
        write("temp_mix.wav", 8000, np.sum(X[0, :, :], axis=0))

        return torch.Tensor(np.concatenate((mfccs, chroma, mel, contrast, tonnetz))).cuda()
