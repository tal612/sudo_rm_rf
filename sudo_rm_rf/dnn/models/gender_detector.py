import os
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import librosa
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os

# torch.set_default_device('cuda:2')

'''ZFNet model for PyTorch
[Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901v3)
'''

import torch
import torch.nn as nn


class ZFnet2(nn.Module):
    def __init__(self, img_channels, num_classes, dropout=0.5):
        super(ZFnet2, self).__init__()
        self.conv1 = nn.Conv2d(img_channels,
                               out_channels=96,
                               kernel_size=7,
                               stride=2)
        self.norm1 = nn.LocalResponseNorm(96)

        self.conv2 = nn.Conv2d(in_channels=96,
                               out_channels=256,
                               kernel_size=5,
                               stride=2)
        self.norm2 = nn.LocalResponseNorm(256)

        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=384,
                               kernel_size=3,
                               stride=1)

        self.conv4 = nn.Conv2d(in_channels=384,
                               out_channels=384,
                               kernel_size=3,
                               stride=1)

        self.conv5 = nn.Conv2d(in_channels=384,
                               out_channels=256,
                               kernel_size=3,
                               stride=1)
        self.norm3 = nn.LocalResponseNorm(256)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()

    def forward(self, X):
        x = extract_mel(X)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def ZFNet2f(img_channels=3, num_classes=1000, dropout=0.5): return ZFnet2(img_channels=img_channels,
                                                                       num_classes=num_classes, dropout=dropout)
class GenderDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(0)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(579, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 3),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(extract_features(x))
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits


class ZFNet1(nn.Module):

    def __init__(self):
        super(ZFNet1, self).__init__()

        # CONV PART.
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1)),
            ('act1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)),
            ('conv2', nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0)),
            ('act2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)),
            ('conv3', nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)),
            ('act3', nn.ReLU()),
            ('conv4', nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)),
            ('act4', nn.ReLU()),
            ('conv5', nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)),
            ('act5', nn.ReLU()),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True))
        ]))

        self.feature_outputs = [0] * len(self.features)
        self.switch_indices = dict()
        self.sizes = dict()

        self.classifier = nn.Sequential(OrderedDict([
            ('fc6', nn.Linear(9216, 4096)),
            ('act6', nn.ReLU()),
            ('fc7', nn.Linear(4096, 4096)),
            ('act7', nn.ReLU()),
            ('fc8', nn.Linear(4096, 3))
        ]))

        # DECONV PART.
        self.deconv_pool5 = nn.MaxUnpool2d(kernel_size=3,
                                           stride=2,
                                           padding=0)
        self.deconv_act5 = nn.ReLU()
        self.deconv_conv5 = nn.ConvTranspose2d(256,
                                               384,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=False)

        self.deconv_act4 = nn.ReLU()
        self.deconv_conv4 = nn.ConvTranspose2d(384,
                                               384,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=False)

        self.deconv_act3 = nn.ReLU()
        self.deconv_conv3 = nn.ConvTranspose2d(384,
                                               256,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=False)

        self.deconv_pool2 = nn.MaxUnpool2d(kernel_size=3,
                                           stride=2,
                                           padding=1)
        self.deconv_act2 = nn.ReLU()
        self.deconv_conv2 = nn.ConvTranspose2d(256,
                                               96,
                                               kernel_size=5,
                                               stride=2,
                                               padding=0,
                                               bias=False)

        self.deconv_pool1 = nn.MaxUnpool2d(kernel_size=3,
                                           stride=2,
                                           padding=1)
        self.deconv_act1 = nn.ReLU()
        self.deconv_conv1 = nn.ConvTranspose2d(96,
                                               3,
                                               kernel_size=7,
                                               stride=2,
                                               padding=1,
                                               bias=False)

    def forward(self, X):
        x = extract_mel(X)
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                self.feature_outputs[i] = x
                self.switch_indices[i] = indices
            else:
                x = layer(x)
                self.feature_outputs[i] = x

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_deconv(self, x, layer):
        if layer < 1 or layer > 5:
            raise Exception("ZFnet -> forward_deconv(): layer value should be between [1, 5]")

        x = self.deconv_pool5(x,
                              self.switch_indices[12],
                              output_size=self.feature_outputs[-2].shape[-2:])
        x = self.deconv_act5(x)
        x = self.deconv_conv5(x)

        if layer == 1:
            return x

        x = self.deconv_act4(x)
        x = self.deconv_conv4(x)

        if layer == 2:
            return x

        x = self.deconv_act3(x)
        x = self.deconv_conv3(x)

        if layer == 3:
            return x

        x = self.deconv_pool2(x,
                              self.switch_indices[5],
                              output_size=self.feature_outputs[4].shape[-2:])
        x = self.deconv_act2(x)
        x = self.deconv_conv2(x)

        if layer == 4:
            return x

        x = self.deconv_pool1(x,
                              self.switch_indices[2],
                              output_size=self.feature_outputs[1].shape[-2:])
        x = self.deconv_act1(x)
        x = self.deconv_conv1(x)

        if layer == 5:
            return x

class ZFNet(nn.Module):
    """Original ZFNET Architecture"""

    def __init__(self, channels=3, class_count=3):
        super(ZFNet, self).__init__()
        self.channels = channels
        self.class_count = class_count

        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()

    def get_conv_net(self):
        layers = []

        # in_channels = self.channels, out_channels = 96
        # kernel_size = 7x7, stride = 2
        layer = nn.Conv2d(
            self.channels, 96, kernel_size=7, stride=2, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))

        # in_channels = 96, out_channels = 256
        # kernel_size = 5x5, stride = 2
        layer = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))

        # in_channels = 256, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))

        # in_channels = 384, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))

        # in_channels = 384, out_channels = 256
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        return nn.Sequential(*layers)

    def get_fc_net(self):
        layers = []

        # in_channels = 9216 -> output of self.conv_net
        # out_channels = 4096
        layer = nn.Linear(9216, 4096)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout())

        # in_channels = 4096
        # out_channels = self.class_count
        layer = nn.Linear(4096, self.class_count)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout())

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        y = self.conv_net(extract_mel(x))
        y = y.view(-1, 9216)
        # print(f'{y=}')
        y = self.fc_net(y)

        return y

def extract_mel(signal):
    y = signal.numpy(force=True)
    y = np.sum(y[0], axis=0)
    # print(f'{y.shape=}')
    sample_rate = 8000
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate
    #                                                        )(signal)
    # print(mel_spectrogram.shape)
    # return mel_spectrogram.cuda()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #

    ms = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    # print(f'{ms.shape=}')
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    img1 = librosa.display.specshow(log_ms, sr=sample_rate, fmax=280,fmin=20, ax=ax)
    # print(f'{img1.shape=}')
    # fig.colorbar(img1, ax=ax)
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img.reshape((224,224,3))
    # fig.clear()
    # plt.axis('off')
    # print(f'{img.shape=}')
    # plt.imshow(img)
    # plt.show()
    # print(f'{img=}')
    return torch.Tensor(img.T).cuda(2)

    # plt.show()
    # return ms
    # log_ms = librosa.power_to_db(ms, ref=np.max)
    # librosa.display.specshow(log_ms, sr=sr)
    #
    # print("CHECK")
    # print(img.shape)
    # return img
def extract_features(signal):
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

    return torch.Tensor(np.concatenate((mfccs, chroma, mel, contrast, tonnetz))).cuda()

    # print(mfccs.shape)
    # print(f"{mfccs.shape=}")
    # print(f"{chroma.shape=}")
    # print(f"{mel.shape=}")
    # print(f"{contrast.shape=}")
    # print(f"{tonnetz.shape=}")
    #
    # os.chdir("/home/dsi/yechezo/sudo_rm_rf_/sudo_rm_rf/temp")
    # # print(X.shape)
    # # print(np.sum(X[0,:,:],axis=0).shape)
    # write("temp1.wav", 8000, X[0, 0, :])
    # write("temp2.wav", 8000, X[0, 1, :])
    # write("temp_mix.wav", 8000, np.sum(X[0, :, :], axis=0))
    #
    # mel = librosa.feature.melspectrogram(y=X, sr=sample_rate).T
    #
    #
    # fig, axis = plt.subplots(1,2, figsize=(12, 6))
    # ax = axis[0]
    #
    # S_dB = librosa.power_to_db(mel[:,:,0,0], ref=np.max)
    # # print(S_dB[:,0])
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                                y_axis='mel', sr=sample_rate, fmin=40,
    #                                fmax=280, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram speaker 1')
    # ax1 = axis[1]
    # # print(mel.shape)
    # S_dB1 = librosa.power_to_db(mel[:, :, 1, 0], ref=np.max)
    # # print(S_dB[:,0])
    # img1 = librosa.display.specshow(S_dB1, x_axis='time',
    #                                y_axis='mel', sr=sample_rate, fmin=40,
    #                                fmax=280, ax=ax1)
    # fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
    # ax1.set(title='Mel-frequency spectrogram speaker 2')
    # plt.show()
    # plt.tight_layout()


