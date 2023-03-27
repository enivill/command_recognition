from src.visualization.plotter import Plotter
import os
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import errno
from tqdm import tqdm
import numpy as np

WORDS = ['bed', 'down', 'happy', 'three', 'tree', 'wow', 'zero']


# TODO
#  We don't need to save the images as .png. We have to extract the features during the training
#  and send these features as arrays directly to the model.
#  So every time we run the training, we can change the feature extraction settings.

# TODO
#  so we need to rewrite this function, don't need Plotter or printing image.
#  We don't need to make image at all. We just need the array.


def feature_extraction_wrong(feature_type: str, data_path: str, files_no: int):
    #     """
    #     create features with Plotter from the given data
    #     save the plots to reports/features/ directory
    #     you can change the audio file's sample rate 'sr_load'
    #     :param files_no: from how many audio files per folder do you want extract features
    #     :param feature_type: stft, mel, mfcc
    #     :param data_path: data path
    #     :return:
    #     """
    #
    #     try:
    #         assert feature_type in ['stft', 'mel', 'mfcc'], "Wrong type. Please use one of these: stft, mel, mfcc."
    #     except Exception as e:
    #         print(e)
    #         return
    #
    #     # changeable options
    #     sr_load = 8000  # sample rate
    #     n_fft = 128
    #     window_length = n_fft
    #     hop_length = window_length // 2
    #     n_mels = 40
    #
    #     fig = Plotter()
    #
    #     for word in tqdm(WORDS):
    #         parent_dir = "reports/features/"
    #         directory = f'{word}_sr{sr_load}_nfft{n_fft}_wl{window_length}_hl{hop_length}'
    #         path = os.path.join(parent_dir, directory)
    #         # if folder already exists - no problem
    #         # other errors - block them
    #         try:
    #             os.mkdir(path)
    #         except OSError as error:
    #             if error.errno != errno.EEXIST:
    #                 raise
    #         print("Directory '% s' created" % directory)
    #
    #         for filename in (os.listdir(f'{data_path}{word}'))[:files_no]:
    #             f = os.path.join(word, filename)
    #
    #             # this returns an audio time series as a numpy array with a default sampling rate of 22KHz mono
    #             # originally our audio has 16000 samples per second,
    #             # resample to 'sr_load' samples per second
    #             y, sr = librosa.load(f'{data_path}{f}', sr=sr_load)
    #
    #             if feature_type == 'stft':
    #                 fig.stft(word, filename, y, n_fft, window_length, hop_length, feature_extr=True)
    #             elif feature_type == 'mel':
    #                 fig.mel(word, filename, y, n_fft, sr_load, window_length, hop_length, n_mels, feature_extr=True)
    #             elif feature_type == 'mfcc':
    #                 fig.mfcc(word, filename, y, n_fft, sr_load, window_length, hop_length, n_mels, feature_extr=True)
    #
    #             # SAVE figure
    #             plt.axis('off')
    #             fig.fig.savefig(
    #                 f'reports/features/{feature_type}/{directory}/{filename}.png', format='png', bbox_inches='tight',
    #                 pad_inches=0)
    #     plt.close()
    pass


# TODO
#  do the same for stft and mfcc
def mel(y: np.ndarray, sr: float) -> np.ndarray:
    # changeable options
    # sr_load = 16000  # sample rate
    n_fft = 128
    window_length = n_fft
    hop_length = window_length // 2
    n_mels = 40
    # fyi
    #  dimension of the mel and mel_decibel is (40, 126)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        n_mels=n_mels
    )
    mel_spectrogram_decibel = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_decibel


def feature_extraction(feature_type: str, data: np.ndarray, root: str) -> np.ndarray:
    features = []
    audio_feature_dict = {}

    for file in tqdm(data):
        if file not in audio_feature_dict:
            y, sr = load_audio(file, root)
            if feature_type == 'mel':
                audio_feature_dict[file] = mel(y, sr)
            if feature_type == 'stft':
                audio_feature_dict[file] = mel(y, sr)
            else:  # mfcc
                audio_feature_dict[file] = mel(y, sr)
        features.append(audio_feature_dict[file])

    return np.array(features, dtype=object)


def fill_in_with_zeros(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    If the audio is shorter than the other ones (in our case 1.0 second) then we fill in the missing space with zeroes
    :param audio: short audio
    :param sample_rate: the length, we want to reach
    :return:
    """
    missing_space = np.zeros((int(sample_rate) - len(audio)))
    audio = np.concatenate((audio, missing_space), axis=0)
    del missing_space
    return audio


def load_audio(data_path: str, root: str) -> (np.ndarray, float):
    # TODO move sr_load to a config file
    sr_load = 8000
    y, sr = librosa.load(f"{root}{data_path}", sr=sr_load)

    # Shorter audios fill in with zeros
    if len(y) < sr_load:
        y = fill_in_with_zeros(y, sr)
    return y, sr


# TODO you can delete this, i just used it for testing
def plot_audio(y: np.ndarray, sr: float):
    """
    plot the audio
    :param y: audio np.ndarray
    :param sr: sample rate
    :return:
    """
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.show()
