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
    def __init__(self, classes):
        super().__init__()
        self.flatten = nn.Flatten(0)
        self.l1_conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=64, stride=2)
        self.l2_relu1 = nn.ReLU()
        self.l3_BN1 = nn.BatchNorm1d(num_features=16)

        self.l4_PL1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.l5_relu2 = nn.ReLU()

        self.l6_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=2)
        self.l7_relu3 = nn.ReLU()
        self.l8_BN2 = nn.BatchNorm1d(num_features=32)

        self.l9_PL2 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.l10_relu4 = nn.ReLU()

        self.l11_conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=2)
        self.l12_relu5 = nn.ReLU()
        self.l13_BN3 = nn.BatchNorm1d(num_features=64)

        self.l14_conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2)
        self.l15_relu6 = nn.ReLU()
        self.l16_BN4 = nn.BatchNorm1d(num_features=128)

        self.l17_conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.l18_relu7 = nn.ReLU()
        self.l19_BN5 = nn.BatchNorm1d(num_features=256)

        self.l20_PL3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.l21_relu8 = nn.ReLU()

        self.FC1 = nn.Linear(in_features=512,out_features=256)
        self.FC1_relu = nn.ReLU()
        self.FC1_dropout = nn.Dropout(p=0.25) # Original = 0.25

        self.FC2 = nn.Linear(in_features=256, out_features=64)
        self.FC2_relu = nn.ReLU()
        self.FC2_dropout = nn.Dropout(p=0.25) # Original = 0.25

        self.FC3 = nn.Linear(in_features=64, out_features=classes)

    def forward(self, _input):
        x = self.l1_conv1(_input)
        x = self.l2_relu1(x)
        x = self.l3_BN1(x)

        x = self.l4_PL1(x)
        x = self.l5_relu2(x)

        x = self.l6_conv2(x)
        x = self.l7_relu3(x)
        x = self.l8_BN2(x)

        x = self.l9_PL2(x)
        x = self.l10_relu4(x)

        x = self.l11_conv3(x)
        x = self.l12_relu5(x)
        x = self.l13_BN3(x)

        x = self.l14_conv4(x)
        x = self.l15_relu6(x)
        x = self.l16_BN4(x)

        x = self.l17_conv5(x)
        x = self.l18_relu7(x)
        x = self.l19_BN5(x)

        x = self.l20_PL3(x)
        x = self.l21_relu8(x)

        x = self.flatten(x)

        x = self.FC1(x)
        x = self.FC1_relu(x)
        x = self.FC1_dropout(x)
        x = self.FC2(x)
        x = self.FC2_relu(x)
        x = self.FC2_dropout(x)
        x = self.FC3(x)
        return x.unsqueeze(0)

    # def forward(self, _input):

    # def extract_features(self, signal):
    #     X = signal.numpy(force=True)
    #     # print(X)
    #     # print(f"input size is {signal.size()}")
    #
    #     sample_rate = 8000
    #     # Sets the name to be the path to where the file is in my computer
    #     mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    #     # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    #     stft = np.abs(librosa.stft(X))
    #     # Computes a chromagram from a waveform or power spectrogram.
    #     chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    #     # Computes a mel-scaled spectrogram.
    #     mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    #     # Computes spectral contrast
    #     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate, fmin=60).T, axis=0)
    #     # Computes the tonal centroid features (tonnetz)
    #     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    #                                               sr=sample_rate, fmin=20).T, axis=0)
    #
    #     os.chdir("/home/dsi/yechezo/sudo_rm_rf_/sudo_rm_rf/temp")
    #     # print(X.shape)
    #     # print(np.sum(X[0,:,:],axis=0).shape)
    #     write("temp1.wav", 8000, X[0, 0, :])
    #     write("temp2.wav", 8000, X[0, 1, :])
    #     write("temp_mix.wav", 8000, np.sum(X[0, :, :], axis=0))
    #
    #     return torch.Tensor(np.concatenate((mfccs, chroma, mel, contrast, tonnetz))).cuda()
