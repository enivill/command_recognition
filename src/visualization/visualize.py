import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa
from src.features.build_features import load_audio


def print_audio_information(
        file_path: str,
        data: np.ndarray,
        sr: int
) -> None:
    """
    print out:
        file path
        the first 10 elements of the data
        audio file shape
        sample rate in Khz
        duration in seconds
    Args:
        :param file_path:  = file path
        :param data:  = audio time series
        :param sr:  = sampling rate of data
        :return: None
    """

    # print(type(data), type(sr))
    print(f'File name: {file_path}')
    print(f'y: {data[:10]}')
    print(f'y shape: {data.shape} {np.shape(data)}')
    print(f'sample rate (KHz): {sr}')
    print(f'Duration of audio files in seconds: {data.shape[0] / sr}')


def save_audio_report(
        data: np.ndarray,
        sample_rate: int,
        path: str
) -> None:
    """
    Save data as audio file
    :param data:
    :param sample_rate:
    :param path: 'new_file.wav' can be saved as WAV, FLAC, OGG and MAT files
    :return:
    """
    sf.write(f'reports/sound/{path}', data, sample_rate)


def plot_audio(file: str, label: str):
    """
    plot the audio wave
    :param file:
    :param label:
    :return:
    """
    y, sr = load_audio(file, 16000)
    plt.figure(figsize=(15, 3))
    plt.title(label)
    plt.subplots_adjust(bottom=.2)
    plt.xlim([0, 1])
    librosa.display.waveshow(y, sr=sr)
    plt.show()

