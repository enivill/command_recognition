from glob import glob
import matplotlib.pylab as plt
from matplotlib import use as mpl_use
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import numpy as np
from src.visualization.plotter import Plotter
import os

import src.visualization.visualize

mpl_use('Qt5Agg')


def print_hi():
    audio_files = glob('data/sample/*/*.wav')
    # playsound(os.path.dirname(__file__) + '/' + audio_files[0])

    # Spectogram
    y_fourier = librosa.stft(y)  # short-time fourier transform
    y_fourier_decibel = librosa.amplitude_to_db(np.abs(y_fourier), ref=np.max)
    print(y_fourier.shape)
    print(y_fourier_decibel.shape)

    # plot the transformed audio data
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(y_fourier_decibel, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Spectogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()

    # # Mel spectogram
    # y_mels = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128*2)
    # print(y_mels.shape)
    # y_mels_decibel = librosa.amplitude_to_db(y_mels, ref=np.max)
    # # plot mel
    # fig, ax = plt.subplots(figsize=(10, 5))
    # img = librosa.display.specshow(y_mels_decibel, x_axis='time', y_axis='log', ax=ax, sr=8000)
    # ax.set_title('Mel spectogram Example', fontsize=20)
    # fig.colorbar(img, ax=ax, format=f'%0.2f')
    # plt.show()


if __name__ == '__main__':
    # src.visualization.visualize.plot_data_and_save(type='spectrogram')
    src.visualization.visualize.plot_data_and_save(type='mel_spectrogram')

    # print_hi()


