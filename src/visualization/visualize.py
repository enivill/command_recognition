import numpy as np
import soundfile as sf
import os
import librosa
from src.visualization.plotter import Plotter


def print_audio_information(
        file_path: str,
        data: np.ndarray,
        sr: int
) -> None:
    """ print out:
        file path
        the first 10 elements of the data
        audio file shape
        sample rate in Khz
        duration in seconds
    Args:
        file_path: str = file path
        data: np.ndarray = audio time series
        sr: int = sampling rate of data
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
    sf.write(f'reports/sound/{path}', data, sample_rate)


def plot_data_and_save(type: str):
    """
        create plots with Plotter from the sample data
        save the plots to /reports directory
        you can change the audio file's sample rate 'sr_load'
    """
    words = ['bed', 'happy', 'three', 'tree', 'zero']
    data_path = 'data/sample/'
    sr_load = 8000

    # for spectrogram
    n_fft = 128
    window_length = n_fft
    hop_length = window_length // 2

    if type == 'spectrogram':
        sub_path = 'spectrograms'
    elif type == 'mel_spectrogram':
        sub_path = 'mel_spectrograms'
    else:
        sub_path = 'wav_plot'


    for word in words[:1]:
        y_word = []
        sr_word = []
        filename_word = []
        for filename in (os.listdir(f'{data_path}{word}'))[:1]:
            f = os.path.join(word, filename)

            # this returns an audio time series as a numpy array with a default sampling rate of 22KHz mono
            # originally our audio has 16000 samples per second,
            # resample to 8000 samples per second
            y, sr = librosa.load(f'{data_path}{f}', sr=sr_load)
            y_word.append(y)
            sr_word.append(sr)
            filename_word.append(filename)
            print_audio_information(f'{data_path}{f}', y, sr)

        fig = Plotter(rows=1, cols=1)

        row = 0
        col = 0
        for idx, y in enumerate(y_word):
            if idx == 2:
                col = 0
                row += 1
            if type == 'spectrogram':
                fig.plot_fourier_spectrogram(word, filename_word[idx], y, n_fft, window_length, hop_length, row, col)
            elif type == 'mel_spectrogram':
                n_mels = 40
                fig.plot_mel_spectrogram(word, filename_word[idx], y, n_fft, sr_load, window_length, hop_length, n_mels, row, col, True)
                # fig.plot_mel_spectrogram(word, filename_word[idx], y, n_fft, sr_load, window_length, hop_length, n_mels,
                #                          row, col, False)
            else:
                fig.plot_audio_waves(word, filename_word[idx], y, sr_word[idx], row, col)
            col += 1

        # save figure
        # fig.fig.savefig(f'reports/figures/{sub_path}/{word}_sr{sr_load}_n{n_fft}_wl{window_length}_hl{hop_length}.svg',
        #                 format='svg', dpi=1200, bbox_inches='tight')
        fig.fig.savefig(f'reports/figures/{sub_path}/mfcc.svg',
                        format='svg', dpi=1200, bbox_inches='tight')
        fig.fig.show()

        y_word.clear()
        sr_word.clear()
        filename_word.clear()


