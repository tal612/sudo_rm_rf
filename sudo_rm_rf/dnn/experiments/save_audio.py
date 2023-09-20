import numpy as np
import torchaudio
import glob
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from pathlib import PurePosixPath
import torch
import os
from pathlib import Path
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt




def save_audio_original(path, name, separated_signals, target, save_path, samplerate):
    target_audio1 = target[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    target_audio2 = target[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy() #sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy() #sample 0 from batch
    #mix_waves = mix_waves[0, :].cpu().detach().numpy()  #sample 0 from batch and 0 from channel
    Path(save_path).mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", path, save_path+name])
    write(f"{save_path}/output_0.wav", samplerate, separated_audio1.astype(np.float32))
    write(f"{save_path}/output_1.wav", samplerate, separated_audio2.astype(np.float32))
    write(f"{save_path}/clean_0.wav", samplerate, target_audio1.astype(np.float32))
    write(f"{save_path}/clean_1.wav", samplerate, target_audio2.astype(np.float32))



def save_audio(separated_signals, original_mix, clean_signals, save_path, samplerate=8000):
    separated_audio1 = separated_signals[0, 0, :].cpu().detach().numpy()  # sample 0 from batch
    separated_audio2 = separated_signals[0, 1, :].cpu().detach().numpy()  # sample 0 from batch
    clean_audio1 = clean_signals[0, 0, :].cpu().detach().numpy()  # sample 0 from batch
    clean_audio2 = clean_signals[0, 1, :].cpu().detach().numpy()  # sample 0 from batch
    mix = original_mix[0].cpu().detach().numpy()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    write(f"{save_path}/output_0.wav", samplerate, separated_audio1.astype(np.float32))
    write(f"{save_path}/output_1.wav", samplerate, separated_audio2.astype(np.float32))
    write(f"{save_path}/clean_0.wav", samplerate, clean_audio1.astype(np.float32))
    write(f"{save_path}/clean_1.wav", samplerate, clean_audio2.astype(np.float32))
    write(f"{save_path}/clean_1.wav", samplerate, clean_audio2.astype(np.float32))
    write(f"{save_path}/mix.wav", samplerate, mix.astype(np.float32))

