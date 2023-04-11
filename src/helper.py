import os
import librosa
from src.models.generators import SiameseGenerator
from librosa import display
import matplotlib.pyplot as plt


def get_audio_duration(audio_file_path: str):
    '''
    Measures the audio file duration in seconds using librosa
    :param audio_file_path:
    :return: audio duration in seconds
    '''
    audio_duration = librosa.get_duration(filename=audio_file_path)
    return audio_duration


def find_dataset_audio_duration_outliers(dataset_path: str, normal_value: float) -> list:
    """
    Iterate through the dataset and find audio files with duration other than normal_value.
    :param normal_value:
    :param dataset_path:
    :return:
    """
    outliers = []

    for directory in os.scandir(dataset_path):
        if directory.is_dir():
            for file in os.scandir(directory):
                if file.name.endswith(".wav"):
                    duration = get_audio_duration(file.path)
                    if duration != normal_value:
                        outliers.append((duration, file.path))
    return outliers


def find_sample_rate_outliers(dataset_path: str, normal_value: int) -> list:

    outliers = []
    for directory in os.scandir(dataset_path):
        if directory.is_dir():
            print(directory.name)
            for file in os.scandir(directory):
                if file.name.endswith(".wav"):
                    sr = librosa.get_samplerate(file.path)
                    print(sr)
                    if sr != normal_value:
                        outliers.append((sr, file.path))
    return outliers


def test_datagen():
    """
    With this function you can test if custom data generator works as expected.
    :return:
    """
    datagen = SiameseGenerator("test", to_fit=False)

    for iter in range(10):
        print(f"Iter = {iter}\n")
        x, y = datagen.next()
        # print('x_a_shape: ', x[0].shape)
        # print('x_b_shape: ', x[1].shape)
        # print('labels: ', y)

        for idx in range(len(x[0])):
            print(f"{idx}: {x[0][idx][0][:3]}")
            print(f"{idx}: {x[1][idx][0][:3]}")
        print(y)
        print("-----------------------")

        for pair in x:
            for img in pair:
                plt.figure()
                display.specshow(
                    data=img,
                    sr=16000,
                    hop_length=64,
                    n_fft=128,
                    win_length=128,
                    x_axis='time',
                    y_axis='mel'
                )
                plt.show()
