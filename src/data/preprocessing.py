import os
import re
import hashlib
import yaml
from tqdm import tqdm
from src.utils import config as my_config
from src.helper import get_audio_duration

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M


def split_data():
    """
    Splits data into training, validation and testing set. You can edit the percentage
    and other configurations in the config.yaml file
    Here we automatically exclude the _background_noise_ folder.
    It calls a function that returns the audio length, and deletes (if it is requested in configs) if it is shorter than the limit (configs)
    Output: train, test and validation .txt files with audio paths
    :param config_file:  path to config yaml file
    :return:
    """
    config = my_config.get_config()
    # -- CONFIG values --
    dataset_root = f"{config['dataset_root']}"
    train_set_name = config['dataset_name']['train']
    val_set_name = config['dataset_name']['val']
    test_set_name = config['dataset_name']['test']
    raw_data_root = config['raw_data_root']
    duration_limit = config['duration_limit']
    delete_outliers_bool = config['delete_outlier']

    validation_set = []
    train_set = []
    test_set = []
    audio_duration = -1

    print(' ----- SPLIT DATA -----')
    for dirs in os.scandir(raw_data_root):
        if dirs.is_dir() and dirs.name != "_background_noise_":
            print(f"Class: {dirs.name}")
            for file in tqdm(os.listdir(dirs)):
                if delete_outliers_bool:
                    audio_duration = get_audio_duration(f"{raw_data_root}{dirs.name}/{file}")
                if audio_duration >= duration_limit or audio_duration == -1:
                    set_name = which_set(file)
                    # file_path = os.path.join(os.path.basename(dirs), os.path.basename(file)).replace("\\", "/")
                    file_path = f"{dirs.name}/{file}"
                    if set_name == 'training':
                        train_set.append(file_path)
                    elif set_name == 'validation':
                        validation_set.append(file_path)
                    else:
                        test_set.append(file_path)

    # make dataset root directory if not exists
    os.makedirs(os.path.dirname(dataset_root), exist_ok=True)

    # save datasets
    with open(f"{dataset_root}{train_set_name}", 'w') as f:
        f.write('\n'.join(train_set))

    with open(f"{dataset_root}{val_set_name}", 'w') as f:
        f.write('\n'.join(validation_set))

    with open(f"{dataset_root}{test_set_name}", 'w') as f:
        f.write('\n'.join(test_set))

    with open(f"{dataset_root}config.yaml", 'w') as f:
        yaml.dump(config, f)

    print(f"Done. Datasets saved to {dataset_root} directory.")


def which_set(filename: str) -> str:
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
    filename: File path of the data sample.
    config_file: configuration path
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """

    config = my_config.get_config()
    # -- CONFIG values --
    val_percentage = config['percentage']['val']
    test_percentage = config['percentage']['test']

    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name = hash_name.encode('utf-8')
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < val_percentage:
        result = 'validation'
    elif percentage_hash < (test_percentage + val_percentage):
        result = 'testing'
    else:
        result = 'training'

    return result
