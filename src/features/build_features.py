import librosa
import librosa.feature
from librosa import display
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from src.utils import config as my_config
from sklearn.preprocessing import StandardScaler
import os


# TODO
#  do the same for stft and mfcc
def mel(y: np.ndarray, sr: float) -> np.ndarray:
    config = my_config.get_config()
    # changeable options
    # sr_load = 16000  # sample rate
    n_fft = config['n_fft']
    window_length = config['window_length']
    hop_length = config['hop_length']
    n_mels = config['n_mels']
    # fyi
    #  dimension of the mel and mel_decibel is (n_mels, 126)

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


def mfcc(y: np.ndarray, sr: float) -> np.ndarray:
    config = my_config.get_config()
    # changeable options
    # sr_load = 16000  # sample rate
    n_fft = config['n_fft']
    window_length = config['window_length']
    hop_length = config['hop_length']
    n_mels = config['n_mels']
    n_mfcc = config['n_mfcc']
    f_min = config['f_min']
    f_max = config['f_max']

    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        hop_length=hop_length,
        win_length=window_length,
        window='hann',
        n_mels=n_mels,
        n_fft=n_fft,
        # n_mfcc=int((2 / 3) * n_mels)
        n_mfcc=n_mfcc,
        fmin=f_min,
        fmax=f_max
    )
    return mfccs


def stft(y: np.ndarray, sr: float):
    config = my_config.get_config()
    # changeable options
    # sr_load = 16000  # sample rate
    n_fft = config['n_fft']
    window_length = config['window_length']
    hop_length = config['hop_length']
    f_min = config['f_min']
    f_max = config['f_max']

    y_fourier = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        window='hann'
    )
    y_fourier_decibel = librosa.amplitude_to_db(np.abs(y_fourier), ref=np.max)

    return y_fourier_decibel


# TODO
#  so maybe i do a little change in here. We use the dictionary to avoid duplicate extractions. But we put every duplicates in the list
#  so it's taking up space needlessly. Maybe we should leave just the dictionary and return that.
def feature_extraction_dataset(feature_type: str, data: np.ndarray, root: str, sample_rate: int) -> np.ndarray:
    features = []
    audio_feature_dict = {}

    for file in tqdm(data):
        if file not in audio_feature_dict:
            y, sr = load_audio(file, sample_rate, root)
            if feature_type == 'mel':
                audio_feature_dict[file] = mel(y, sr)
            elif feature_type == 'stft':
                audio_feature_dict[file] = mel(y, sr)
            else:  # mfcc
                audio_feature_dict[file] = mfcc(y, sr)
        features.append(audio_feature_dict[file])

    return np.array(features, dtype=object)


def feature_extraction(file: str) -> np.ndarray:
    config = my_config.get_config()

    y, sr = load_audio(file, config['sample_rate'], config['raw_data_root'])
    if config['feature_type'] == 'mel':
        audio_feature = mel(y, sr)
    elif config['feature_type'] == 'stft':
        audio_feature = stft(y, sr)
    else:  # mfcc
        audio_feature = mfcc(y, sr)

    return np.array(audio_feature)


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


def load_audio(data_path: str, sr: int, raw_data_root: str) -> (np.ndarray, float):
    """
    Load an audio file as a floating point time series.
    Audio will be automatically resampled to the given rate (default sr=22050).
    To preserve the native sampling rate of the file, use sr=None.
    :param data_path:
    :param config:
    :return:
    """
    y, sr = librosa.load(f"{raw_data_root}{data_path}", sr=sr)
    # Shorter audios fill in with zeros
    if len(y) < sr:
        y = fill_in_with_zeros(y, sr)
    return y, sr


# TODO you can delete this, I just used it for testing
def plot_audio(y: np.ndarray, label: float, sr: float):
    """
    plot the audio wave
    :param y: audio np.ndarray
    :param sr: sample rate
    :return:
    """
    plt.figure()
    plt.title(f"class label: {label}")
    librosa.display.waveshow(y, sr=sr)
    plt.show()


def plot_mel(y: np.ndarray):
    config = my_config.get_config()

    plt.ioff()
    fig = plt.figure()
    display.specshow(
        data=y,
        sr=config["sample_rate"],
        hop_length=config['hop_length'],
        n_fft=config['n_fft'],
        win_length=config['window_length'],
        x_axis='time',
        y_axis='mel'
    )
    title = f"sr{config['sample_rate']}_hl{config['hop_length']}_nfft{config['n_fft']}_wl{config['window_length']}_mels{config['n_mels']}_mfcc{config['n_mfcc']}_fmin{config['f_min']}_fmax{config['f_max']}_w{y.shape[1]}_h{y.shape[0]}"
    plt.title(title)
    save_path = 'reports/figures/mel_spectrograms/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(f"{save_path}{title}.png")
    plt.close(fig)


def plot_mfcc(mfccs: np.ndarray, norm: bool=False):
    config = my_config.get_config()
    plt.ioff()
    fig = plt.figure()
    if norm:
        scaler = StandardScaler()
        ee = scaler.fit_transform(mfccs.T)

        librosa.display.specshow(ee.T)

    else:
        librosa.display.specshow(
            data=mfccs,
            sr=config["sample_rate"],
            hop_length=config['hop_length'],
            n_fft=config['n_fft'],
            win_length=config['window_length'],
            x_axis='time'
        )

    title = f"sr{config['sample_rate']}_hl{config['hop_length']}_nfft{config['n_fft']}_wl{config['window_length']}_mels{config['n_mels']}_mfcc{config['n_mfcc']}_fmin{config['f_min']}_fmax{config['f_max']}_w{mfccs.shape[1]}_h{mfccs.shape[0]}"
    plt.title(title)
    save_path = 'reports/figures/mfcc_spectrograms/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(f"{save_path}{title}.png")
    plt.close(fig)


def plot_stft(y: np.ndarray):
    config = my_config.get_config()
    plt.ioff()
    fig = plt.figure()
    img = librosa.display.specshow(
        data=y,
        sr=config['sample_rate'],
        hop_length=config['hop_length'],
        n_fft=config['n_fft'],
        win_length=config['window_length'],
        x_axis='time',
        y_axis='log'
    )
    title = f"sr{config['sample_rate']}_hl{config['hop_length']}_nfft{config['n_fft']}_wl{config['window_length']}_w{y.shape[1]}_h{y.shape[0]}"
    plt.title(title)
    save_path = 'reports/figures/spectrograms/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(f"{save_path}{title}.png")
    plt.close(fig)
