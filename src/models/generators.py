from pandas import read_csv
from src.features.build_features import feature_extraction
from sklearn.utils import shuffle
import numpy as np
from src.utils import config as my_config
from keras.api._v2.keras.utils import Sequence


def ceil_div(a, b):
    return -(a // -b)


# class DataGenerator(object):
#     def __init__(self, dataset_name: str, shuffle_data: bool):
#         # config data
#         self.config = my_config.get_config()
#         self.batch_size = self.config["batch_size"]
#         self.raw_data_root = self.config["raw_data_root"]
#         # read dataset
#         self.data = read_csv(f'{self.config["pairs_root"]}{self.config["pairs_name"][dataset_name]}', delimiter=';')
#         self.data = self.data.to_numpy()
#         self.shuffle = shuffle_data
#         if self.shuffle:
#             self.data = shuffle(self.data)
#
#         self.samples_per_data = ceil_div(self.data.shape[0], self.batch_size)
#
#         self.num_data = len(self.data)
#
#     def next(self):
#         """
#         Yields the next training batch.
#         Suppose `samples` is an array [[image1_filename, image2_filename, label1], [image3_filename, image4_filename, label2],...].
#         """
#         while True:  # Loop forever so the generator never terminates
#             # shuffle the data after every iteration
#             if self.shuffle:
#                 self.data = shuffle(self.data)
#             # Get index to start each batch: [0, batch_size, 2*batch_size, ...,
#             # max multiple of batch_size <= num_samples]
#             for offset in range(0, self.num_data, self.batch_size):
#                 # Get the samples you'll use in this batch
#                 batch_samples = self.data[offset:offset + self.batch_size]
#                 X_a = []
#                 X_b = []
#                 y = []
#
#                 for batch_sample in batch_samples:
#                     audio_1_name = batch_sample[0]
#                     audio_2_name = batch_sample[1]
#                     label = batch_sample[2]
#                     # feature extraction
#                     audio_1 = feature_extraction(file=audio_1_name, config=self.config)
#                     audio_2 = feature_extraction(file=audio_2_name, config=self.config)
#                     # Add to arrays
#                     X_a.append(audio_1)
#                     X_b.append(audio_2)
#                     y.append(label)
#
#                 # Make sure they're numpy arrays (as opposed to lists)
#                 X_a = np.array(X_a)
#                 X_b = np.array(X_b)
#                 y = np.array(y).astype('float32')
#
#                 yield [X_a, X_b], y


class SiameseGenerator(Sequence):

    def __init__(self, dataset_name: str, shuffle=True, to_fit=True):

        # Initialization
        self.config = my_config.get_config()
        self.batch_size = self.config["batch_size"]
        self.raw_data_root = self.config["raw_data_root"]
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.data = read_csv(f'{self.config["pairs_root"]}{self.config["pairs_name"][dataset_name]}', delimiter=';')
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

            # feature extraction
            audio_1 = feature_extraction(file=audio_1_name)
            audio_2 = feature_extraction(file=audio_2_name)
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


