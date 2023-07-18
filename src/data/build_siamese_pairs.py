# source https://www.youtube.com/watch?v=jOY-topQ2_c

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter
import yaml
from src.utils import config as my_config


def make_pairs() -> None:
    """
    This function is useful for building image pairs for siamese network
    It reads datasets, calls function to reduce its size if requested (word_per_class_train)
    and makes pairs for siamese network
    You can edit the configurations in the config.yaml file
    :return:
    """
    make_pairs_config = my_config.get_config()['make_pairs']
    paths_config = my_config.get_config()['paths']
    # -- CONFIG values --
    data_path = paths_config['raw_data_root']
    dataset_root = paths_config['dataset_root']
    dataset_name_dict = paths_config['dataset_name']
    excluded_classes_dict = make_pairs_config['excluded_classes']
    word_per_class_train = make_pairs_config['word_per_class_train']
    order_lst = make_pairs_config['order']
    percentage_dict = make_pairs_config['percentage']
    pairs_root = paths_config['pairs_root']
    pairs_name_dict = paths_config['pairs_name']

    # make dataset root directory if not exists
    os.makedirs(os.path.dirname(pairs_root), exist_ok=True)

    for current_set in order_lst:

        # get classes and exclude classes
        classes = get_classes(data_path, exclude=excluded_classes_dict[current_set])
        # classes : {'bed': 0, 'bird': 1, 'cat': 2, ... 'zero': 29}

        # extract files and labels from .txt file to np.arrays
        files, labels = extract_file_names_labels(f"{dataset_root}{dataset_name_dict[current_set]}", classes)
        # files : [bed/00176480_nohash_0.wav, bed/004ae714_nohash_0.wav, ...]
        # labels : [0, 0, 0, 0, 0, ... 29, 29, 29]

        if word_per_class_train is not None:
            # calculate words per class
            if current_set == 'train':
                word_per_class = word_per_class_train
            else:
                word_per_class = round(word_per_class_train/percentage_dict['train']*percentage_dict[current_set])
            print(f"\n ----- REDUCE {current_set} SET, percent: {percentage_dict[current_set]}, words per class: {word_per_class} -----")
            files, labels = reduce_class_files(files, labels, word_per_class=word_per_class)

        # initialize two empty lists to hold the (image, image) pairs and
        # labels to indicate if a pair is positive or negative
        pair_images = []  # = [(image, image), (image, image), (image, image), ...]
        pair_labels = []  # = [[0], [1], [1], [0], [1],...] where 0 - negative pair, 1 - positive pair

        # calculate the total number of classes present in the dataset
        # and then build a list of indexes for each class label that
        # provides the indexes for all examples with a given label
        num_classes = len(np.unique(labels))  # num_classes: 30
        # returns a list of lists, each list represents a class, which contains
        # the image/label indexes that belongs to the class
        idx = [np.where(labels == i)[0] for i in range(0, num_classes)]

        print(f"\n ----- MAKING {current_set} PAIRS ----- ")
        # loop over all files
        for idx_a in tqdm(range(len(files))):
            # grab the current image and label belonging to the current
            # iteration
            current_image = files[idx_a]
            label = labels[idx_a]

            # randomly pick an image that belongs to the *same* class
            # label
            idx_b = np.random.choice([i for i in idx[label] if i != idx_a])
            pos_image = files[idx_b]

            # prepare a positive pair and update the files and labels
            # lists, respectively
            pair_images.append([current_image, pos_image])
            pair_labels.append([1])

            # grab the indices for each of the class labels *not* equal to
            # the current label and randomly pick an image corresponding
            # to a label *not* equal to the current label
            neg_idx = np.where(labels != label)[0]
            neg_image = files[np.random.choice(neg_idx)]

            # prepare a negative pair of files and update our lists
            pair_images.append([current_image, neg_image])
            pair_labels.append([0])

        # return a 2-tuple of our image pairs and labels
        save_pairs_csv(np.array(pair_images), np.array(pair_labels), f'{pairs_root}{pairs_name_dict[current_set]}')

        # save config
        with open(f"{pairs_root}config.yaml", 'w') as f:
            yaml.dump(make_pairs_config, f)

        print(f"\nPair creation is Done. It is saved to {pairs_root}{pairs_name_dict[current_set]}")


def save_pairs_csv(pairs: np.array(tuple), labels: np.array(str), path: str) -> None:
    """
    Save pairs and their labels to csv
    pairs [("bed/00176480_nohash_0.wav" , "bed/590750e8_nohash_0.wav"), ("bed/..", "bird/.."), ("bed/..", "bed/..")]
    labels [1, 0, 1]
    csv
        audio_1;audio_2;label
        bed/00176480_nohash_0.wav;bed/590750e8_nohash_0.wav;1
        bed/00176480_nohash_0.wav;bird/48a9f771_nohash_0.wav;0
        bed/004ae714_nohash_0.wav;bed/4abb2400_nohash_1.wav;1
    :param pairs: numpy array of tuples
    :param labels: numpy array
    :param path: location of saving
    :return:
    """
    pairs_labels = np.column_stack((pairs, labels))
    df = pd.DataFrame(pairs_labels)
    df.to_csv(path, index=False, header=['audio_1', 'audio_2', 'label'], sep=";")


def get_classes(directory: str, exclude: list = None) -> {}:
    """
    Get all sub folder names from a directory - these will be our classes
    :return: dictionary {'bed':0, 'bird':1, 'cat':2,...}
    """
    if exclude is None:
        exclude = []
    sub_folders = [f.name for f in os.scandir(directory) if f.is_dir() and f.name not in exclude]
    sub_folders_dict = {k: v for v, k in enumerate(sub_folders)}
    return sub_folders_dict


def extract_file_names_labels(txt_file: str, classes: {}) -> (np.ndarray, np.ndarray):
    """
    Txt file contains audio file paths in format: word/hash_nohash_0.wav
    This function extracts the class name (the word) and creates x nad y arrays.
    X - string, the audio file name, Y - str, the class name
    :param txt_file:
    :param classes:
    :return:
    """
    x = []
    y = []
    with open(txt_file, 'r') as f:
        file_names = f.read()
        file_name_list = file_names.split("\n")
        for file in file_name_list:
            if file != "":
                class_name, file_name = file.split("/")
                # excluded classes
                if classes.get(class_name) is not None:
                    y.append(classes.get(class_name))
                    x.append(class_name + "/" + file_name)
    return np.array(x), np.array(y)


def reduce_class_files(files: np. ndarray, labels: np.ndarray, word_per_class: int) -> (np.ndarray, np.ndarray):
    """
    Prerequisites for functioning well: files and labels must be sorted by class labels
    Reduces the dataset, each class will contain word_per_class files
    :param word_per_class:
    :param files:
    :param labels:
    :return:
    """
    new_files = []
    new_labels = []

    current_class = int()
    count = 0

    for idx, label in enumerate(labels):
        if label != current_class:
            current_class = label
            count = 0
        if count < word_per_class:
            new_files.append(files[idx])
            new_labels.append(labels[idx])
            count += 1

    print("Classes with less file:")
    counter = Counter(new_labels)
    [print(f'\t{w}: {counter[w]}') for w in counter if counter[w] != word_per_class]

    return np.array(new_files), np.array(new_labels)






