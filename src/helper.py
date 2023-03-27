import re
import os
import numpy as np
import librosa
import yaml

CONFIG_PATH = 'src/configs/'


# Function to load yaml configuration file
def load_config(config_name: str):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


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
