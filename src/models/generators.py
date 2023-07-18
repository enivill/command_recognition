from keras.utils import Sequence
from pandas import read_csv
import numpy as np
from src.utils import config as my_config

import os


def ceil_div(a, b):
    return -(a // -b)


class SiameseGenerator(Sequence):

    def __init__(self, dataset_name: str, shuffle=True, to_fit=True):

        # Initialization
        self.paths_config = my_config.get_config()['paths']
        self.train_config = my_config.get_config()['train']
        self.batch_size =self.train_config["batch_size"]
        self.raw_data_root = self.paths_config["raw_data_root"]
        self.features_path = self.paths_config["features_path"]
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.data = read_csv(f'{self.paths_config["pairs_root"]}{self.paths_config["pairs_name"][dataset_name]}', delimiter=';')
        self.data = self.data.to_numpy()
        self.datalen = len(self.data)
        self.indexes = np.arange(self.datalen)
        self.n = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # def __next__(self):
    #     # Get one batch of data
    #     data = self.__getitem__(self.n)
    #     # Batch index
    #     self.n += 1
    #
    #     # If we have processed the entire dataset then
    #     if self.n >= self.__len__():
    #         self.on_epoch_end()
    #         self.n = 0
    #
    #     return data

    def __getitem__(self, index):
        """Generate one batch of data
            :param index: index of the batch
            :return: X and y when fitting. X only when predicting
        """
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_samples = self.data[batch_indexes]

        X_a = []
        X_b = []
        y = []

        for batch_sample in batch_samples:
            audio_1_name = batch_sample[0]
            audio_2_name = batch_sample[1]

            # load feature from file
            split_name_1 = os.path.split(audio_1_name)
            audio_1_name_no_ext = os.path.splitext(split_name_1[-1])[0]
            split_name_2 = os.path.split(audio_2_name)
            audio_2_name_no_ext = os.path.splitext(split_name_2[-1])[0]
            suffix = '.npy'
            path_a = os.path.join(self.features_path, split_name_1[0])
            path_b = os.path.join(self.features_path, split_name_2[0])
            audio_1 = np.load(os.path.join(path_a, audio_1_name_no_ext + suffix))
            audio_2 = np.load(os.path.join(path_b, audio_2_name_no_ext + suffix))

            # feature extraction
            # audio_1 = feature_extraction(file=audio_1_name)
            # audio_2 = feature_extraction(file=audio_2_name)

            # Add to arrays
            X_a.append(audio_1)
            X_b.append(audio_2)

            if self.to_fit:
                label = batch_sample[2]
                y.append(label)

        # Make sure they're numpy arrays (as opposed to lists)
        X_a = np.array(X_a)
        X_b = np.array(X_b)

        if self.to_fit:
            y = np.array(y).astype('float32')
            return [X_a, X_b], y
        else:
            return [X_a, X_b], y

    def __len__(self):
        # Denotes the number of batches per epoch
        return ceil_div(self.datalen, self.batch_size)

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)


