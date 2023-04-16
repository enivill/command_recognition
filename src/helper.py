import os
import librosa
from src.models.generators import SiameseGenerator
from librosa import display
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm



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
                    if sr != normal_value:
                        outliers.append((sr, file.path))
    return outliers


def _test_datagen():
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


def distribution_of_classes():
    """
    First figure plots the count of each word in the dataset.
    The second figure plots the audio lengths.
        There are 3 bars, in the first one is the number of audios which have shorter sample lengths than 1 second,
        the second bar shows the number of audios, which has exactly 1 second length,
        the 3rd bar are audios, which are longer than 1 sec.
    The length is set to 16000, because we work with the sample rate of 16kHz. 16000/16000 = 1 sec.
    If a sample has for example 14500 sample length, it means that its length in seconds is 54.375 .
    14500/16000=0.90625; 60*0.90625 = 54.375
    :return:
    """
    dataset_path = "data/external/speech_commands_v0.01"
    train_labels = []
    train_samples = []
    # Loading all the waves and labels
    for directory in os.scandir(dataset_path):
        if directory.is_dir() and directory.name != '_background_noise_':
            print(f"Processing class: {directory.name}")
            for wav_file in os.scandir(directory):
                if wav_file.name.endswith(".wav"):
                    sample, sample_rate = librosa.load(wav_file.path, sr=16000)
                    train_labels.append(directory.name)
                    train_samples.append(len(sample))

    train_pd = pd.DataFrame({'label': train_labels})
    counts = train_pd["label"].value_counts()
    fig = plt.figure(figsize=(15, 20))
    plt.barh(y=counts.index, width=counts.values)
    plt.xlabel("počet", fontsize=20)
    plt.ylabel("slová", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks([i for i in range(1600, 2500) if i % 100 == 0], fontsize=20)
    plt.xlim([1300, 2500])
    plt.subplots_adjust(bottom=.1, left=.2)
    plt.grid(visible=True, axis='x', color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.show()

    good_length = 0
    less_length = 0
    more_length = 0

    for length in train_samples:
        if length == 16000:
            good_length += 1
        elif length < 16000:
            less_length += 1
        else:
             more_length += 1

    x = [less_length, good_length, more_length]
    labels = ['menej ako 1 sekunda', '1 sekunda', 'viac ako 1 sekunda']
    print(f"Samples with one second length: {good_length}")
    print(f"Samples with less than one second length: {less_length}")
    print(f"Samples with more than one second length: {more_length}")
    fig = plt.figure(figsize=(15, 15))
    plt.bar(x=labels, height=x)
    plt.xlabel("dĺžka vzoriek", fontsize=20)
    plt.ylabel("počet vzoriek", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()
