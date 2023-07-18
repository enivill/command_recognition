import librosa
import librosa.feature
from librosa import display
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from src.utils import config as my_config
from sklearn.preprocessing import StandardScaler
import os


def mel(y: np.ndarray, sr: float) -> np.ndarray:
    config = my_config.get_config()['feature_extraction']

    # changeable options
    # sr_load = 16000  # sample rate
    n_fft, window_length, hop_length = calculate_nfft_wl_hl()
    n_mels = config['n_mels']
    f_min = config['f_min']
    f_max = config['f_max']

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )
    mel_spectrogram_decibel = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_decibel


def mfcc(y: np.ndarray, sr: float) -> np.ndarray:
    config = my_config.get_config()['feature_extraction']

    n_fft, window_length, hop_length = calculate_nfft_wl_hl()
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
        n_mfcc=n_mfcc,
        fmin=f_min,
        fmax=f_max
    )
    return mfccs


def stft(y: np.ndarray):
    n_fft, window_length, hop_length = calculate_nfft_wl_hl()

    y_fourier = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        window='hann'
    )
    y_fourier_decibel = librosa.amplitude_to_db(np.abs(y_fourier), ref=np.max)

    return y_fourier_decibel



def feature_extraction_dataset():
    """
    Doing feature extraction on the whole dataset, saves data as .npy file to config['paths']['features_path']/word/filename
    you must specify the feature_type variable in config file.
    """
    config = my_config.get_config()

    if not os.path.exists(config['paths']['features_path']):
        os.mkdir(config['paths']['features_path'])

    for dirs in os.scandir(config['paths']['raw_data_root']):
        if dirs.is_dir() and dirs.name != "_background_noise_":
            print(f"Class: {dirs.name}")
            for file in tqdm(os.listdir(dirs)):
                file_path = f"{dirs.name}/{file}"
                feature = feature_extraction(file_path)
                path_split = os.path.split(file_path)
                file_name = os.path.splitext(path_split[-1])[0]
                path_a = os.path.join(config['paths']['features_path'], path_split[0])
                if not os.path.exists(path_a):
                    os.mkdir(path_a)
                np.save(os.path.join(path_a, file_name), feature)


def feature_extraction(file: str) -> np.ndarray:
    """
    Extracts features from audio file
    features types can be: mel, stft, mfcc
    you must specify the feature type in the config file
    """
    config = my_config.get_config()

    y, sr = load_audio(os.path.join(config['paths']['raw_data_root'], file),
                       config['feature_extraction']['sample_rate'])
    if config['train']['feature_type'] == 'mel':
        audio_feature = mel(y, sr)
    elif config['train']['feature_type'] == 'stft':
        audio_feature = stft(y)
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


def load_audio(data_path: str, sr: int) -> (np.ndarray, float):
    """
    Load an audio file as a floating point time series.
    Audio will be automatically resampled to the given rate (default sr=22050).
    To preserve the native sampling rate of the file, use sr=None.
    :param sr:
    :param data_path:
    :return:
    """
    y, sr = librosa.load(data_path, sr=sr)
    # Shorter audios fill in with zeros
    if len(y) < sr:
        y = fill_in_with_zeros(y, sr)
    return y, sr


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


def plot_mel(y: np.ndarray) -> None:
    config = my_config.get_config()['feature_extraction']
    n_fft, window_length, hop_length = calculate_nfft_wl_hl()

    plt.ioff()
    fig = plt.figure()
    display.specshow(
        data=y,
        sr=config["sample_rate"],
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=window_length,
        fmin=config['f_min'],
        fmax=config['f_max'],
        x_axis='frames',
        y_axis='mel'
    )
    plt.show()
    # title = f"sr{config['sample_rate']}_hl{hop_length}_nfft{n_fft}_wl{window_length}_mels{config['n_mels']}_fmin{config['f_min']}_fmax{config['f_max']}_w{y.shape[1]}_h{y.shape[0]}"
    # plt.title(title)
    # save_path = 'reports/figures/mel_spectrograms/'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f"{save_path}{title}.png")
    # plt.close(fig)


def plot_mfcc(mfccs: np.ndarray, norm: bool = False):
    config = my_config.get_config()['feature_extraction']
    n_fft, window_length, hop_length = calculate_nfft_wl_hl()

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
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=window_length,
            x_axis='time'
        )

    title = f"sr{config['sample_rate']}_hl{hop_length}_nfft{n_fft}_wl{window_length}_mels{config['n_mels']}_mfcc{config['n_mfcc']}_fmin{config['f_min']}_fmax{config['f_max']}_w{mfccs.shape[1]}_h{mfccs.shape[0]}"
    plt.title(title)
    save_path = 'reports/figures/mfcc_spectrograms/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(f"{save_path}{title}.png")
    plt.close(fig)


def plot_stft(y: np.ndarray):
    config = my_config.get_config()['feature_extraction']
    n_fft, window_length, hop_length = calculate_nfft_wl_hl()

    plt.ioff()
    fig = plt.figure()
    img = librosa.display.specshow(
        data=y,
        sr=config['sample_rate'],
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=window_length,
        # x_axis='time',
        # y_axis='log'
        x_axis='frames',
        y_axis='hz'
    )
    title = f"sr{config['sample_rate']}_hl{hop_length}_nfft{n_fft}_wl{window_length}_w{y.shape[1]}_h{y.shape[0]}"
    plt.title(title)
    save_path = 'reports/figures/spectrograms/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(f"{save_path}{title}.png")
    plt.close(fig)


def calculate_nfft_wl_hl() -> (int, int, int):
    """
    Calculates the n_fft, window_length and hop_length from given wl and hl values in seconds and the sample rate.
    :return:
    """
    config = my_config.get_config()['feature_extraction']
    window_length_seconds = config['window_length_seconds']
    hop_length_seconds = config['hop_length_seconds']
    sample_rate = config['sample_rate']

    window_length = window_length_seconds / 1000 * sample_rate
    hop_length = hop_length_seconds / 1000 * sample_rate
    nfft = next_power_of_2(int(window_length))

    return int(nfft), int(window_length), int(hop_length)


def next_power_of_2(x: int):
    """
    Finds the smallest power of 2 greater than or equal to x
    :param x:
    :return:
    """
    return 1 << (x - 1).bit_length()
