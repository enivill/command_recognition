from src.visualization.plotter import Plotter
import os
import librosa
import matplotlib.pyplot as plt
import errno
from tqdm import tqdm

WORDS = ['bed', 'down', 'happy', 'three', 'tree', 'wow', 'zero']


def feature_extraction(feature_type: str, data_path: str, files_no: int):
    """
    create features with Plotter from the given data
    save the plots to reports/features/ directory
    you can change the audio file's sample rate 'sr_load'
    :param files_no: from how many audio files per folder do you want extract features
    :param feature_type: stft, mel, mfcc
    :param data_path: data path
    :return:
    """

    try:
        assert feature_type in ['stft', 'mel', 'mfcc'], "Wrong type. Please use one of these: stft, mel, mfcc."
    except Exception as e:
        print(e)
        return

    # changeable options
    sr_load = 8000  # sample rate
    n_fft = 128
    window_length = n_fft
    hop_length = window_length // 2
    n_mels = 40

    fig = Plotter()

    for word in tqdm(WORDS):
        parent_dir = "reports/features/"
        directory = f'{word}_sr{sr_load}_nfft{n_fft}_wl{window_length}_hl{hop_length}'
        path = os.path.join(parent_dir, directory)
        # if folder already exists - no problem
        # other errors - block them
        try:
            os.mkdir(path)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
        print("Directory '% s' created" % directory)

        for filename in (os.listdir(f'{data_path}{word}'))[:files_no]:
            f = os.path.join(word, filename)

            # this returns an audio time series as a numpy array with a default sampling rate of 22KHz mono
            # originally our audio has 16000 samples per second,
            # resample to 'sr_load' samples per second
            y, sr = librosa.load(f'{data_path}{f}', sr=sr_load)

            if feature_type == 'stft':
                fig.stft(word, filename, y, n_fft, window_length, hop_length, feature_extr=True)
            elif feature_type == 'mel':
                fig.mel(word, filename, y, n_fft, sr_load, window_length, hop_length, n_mels, feature_extr=True)
            elif feature_type == 'mfcc':
                fig.mfcc(word, filename, y, n_fft, sr_load, window_length, hop_length, n_mels, feature_extr=True)

            # SAVE figure
            plt.axis('off')
            fig.fig.savefig(
                f'reports/features/{feature_type}/{directory}/{filename}.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

