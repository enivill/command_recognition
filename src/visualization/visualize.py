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


def plot_data_and_save(_type: str, words_no: int = None, files_no: int = None):
    """
    create plots with Plotter from the sample data
    save the plots to /reports directory
    you can change the audio file's sample rate 'sr_load'
    :param words_no: how many words from data, default None - till the end of the list
    :param files_no: how many files from word dictionary, default None - till the end of the list
    :param _type: stft, mel, mfcc or wave
    :return:
    """
    type_path = {'stft': 'spectrograms', 'mel': 'mel_spectrograms', 'mfcc': 'mfcc_spectrograms', 'wave': 'wav_plot'}
    try:
        assert _type in type_path.keys(), "Wrong type. Please use one of these: wave, stft, mel, mfcc."
    except Exception as e:
        print(e)
        return

    words = [x for x in os.listdir('data/sample') if x != '.gitignore']
    data_path = 'data/sample/'

    # for spectrogram
    sr_load = 8000  # sample rate
    n_fft = 128
    window_length = n_fft
    hop_length = window_length // 2

    for word in words[:words_no]:
        y_word = []
        sr_word = []
        filename_word = []
        for filename in (os.listdir(f'{data_path}{word}'))[:files_no]:
            f = os.path.join(word, filename)

            # this returns an audio time series as a numpy array with a default sampling rate of 22KHz mono
            # originally our audio has 16000 samples per second,
            # resample to 'sr_load' samples per second
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
            if _type == 'stft':
                fig.stft(word, filename_word[idx], y, n_fft, window_length, hop_length, row, col)
            elif _type == 'mel':
                n_mels = 40
                fig.mel(word, filename_word[idx], y, n_fft, sr_load, window_length, hop_length, n_mels,
                        row, col)
            elif _type == 'mfcc':
                n_mels = 40
                fig.mfcc(word, filename_word[idx], y, n_fft, sr_load, window_length, hop_length, n_mels,
                         row, col)
            else:
                fig.plot_audio_waves(word, filename_word[idx], y, sr_word[idx], row, col)
            col += 1

        # SAVE figure
        if _type == '':
            fig.fig.savefig(
                f'reports/figures/{type_path[_type]}/{word}_sr{sr_load}.svg',
                format='svg', dpi=1200, bbox_inches='tight')
        else:
            fig.fig.savefig(
                f'reports/figures/{type_path[_type]}/{word}_sr{sr_load}_n{n_fft}_wl{window_length}_hl{hop_length}.svg',
                format='svg', dpi=1200, bbox_inches='tight')

        fig.fig.show()

        # clear lists
        y_word.clear()
        sr_word.clear()
        filename_word.clear()
