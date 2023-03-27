import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(1337)  # for reproducibility

import matplotlib.pyplot as plt
import random
from keras.api._v2.keras.datasets import mnist
from keras.api._v2.keras.models import Sequential, Model
from keras.api._v2.keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D
from keras.api._v2.keras.optimizers import SGD, RMSprop
from keras.api._v2.keras import backend as K
from keras.api._v2.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, Callback
from pandas import read_csv
import tensorflow as tf
from src.features.build_features import feature_extraction
from tensorflow.python.client import device_lib
import os
import json
from keras.api._v2.keras.models import model_from_json
from sklearn.decomposition import PCA


TRAIN_NUM = 20000
VAL_NUM = 2000


class DataGenerator(object):
    def __init__(self, batch_size):
        # train data
        train_pairs_labels = read_csv("reports/pairs.csv", delimiter=';')
        tr_pairs_0_paths = train_pairs_labels['audio_1'].to_numpy()#[:TRAIN_NUM]
        tr_pairs_1_paths = train_pairs_labels['audio_2'].to_numpy()#[:TRAIN_NUM]
        self.tr_y = train_pairs_labels['label'].to_numpy().astype('float32')#[:TRAIN_NUM].astype('float32')
        global TRAIN_NUM
        TRAIN_NUM = len(tr_pairs_0_paths)
        # TODO make congfig file for : root
        print("-----FEATURE EXTRACTION-----")
        print("training set pairs 1/2...")
        tr_pairs_0 = feature_extraction('mel', tr_pairs_0_paths, "data/external/speech_commands_v0.01/")
        print("training set pairs 2/2...")
        tr_pairs_1 = feature_extraction('mel', tr_pairs_1_paths, "data/external/speech_commands_v0.01/")

        # validation data
        val_pairs_labels = read_csv("reports/val_pairs.csv", delimiter=';')
        val_pairs_0_paths = val_pairs_labels['audio_1'].to_numpy()#[:VAL_NUM]
        val_pairs_1_paths = val_pairs_labels['audio_2'].to_numpy()#[:VAL_NUM]
        self.val_y = val_pairs_labels['label'].to_numpy().astype('float32')#[:VAL_NUM].astype('float32')
        global VAL_NUM
        VAL_NUM = len(val_pairs_0_paths)
        # TODO make congfig file for : root
        print("validation set pairs 1/2...")
        val_pairs_0 = feature_extraction('mel', val_pairs_0_paths, "data/external/speech_commands_v0.01/")
        print("validation set pairs 2/2...")
        val_pairs_1 = feature_extraction('mel', val_pairs_1_paths, "data/external/speech_commands_v0.01/")
        print("FEATURE EXTRACTION Done.")

        # feature height and width
        tr_w = tr_pairs_0.shape[1]
        tr_h = tr_pairs_0.shape[2]
        tr_w_h = tr_w * tr_h
        # feature height and width
        val_w = val_pairs_0.shape[1]
        val_h = val_pairs_0.shape[2]
        val_w_h = val_w * val_h

        # reshape data from 3D to 2D array
        # self.tr_pairs_0 = np.reshape(tr_pairs_0, (tr_pairs_0.shape[0], tr_w_h)).astype('float32')
        # self.tr_pairs_1 = np.reshape(tr_pairs_1, (tr_pairs_1.shape[0], tr_w_h)).astype('float32')
        # self.val_pairs_0 = np.reshape(val_pairs_0, (val_pairs_0.shape[0], val_w_h)).astype('float32')
        # self.val_pairs_1 = np.reshape(val_pairs_1, (val_pairs_1.shape[0], val_w_h)).astype('float32')
        #
        self.tr_pairs_0 = np.reshape(tr_pairs_0, list(tr_pairs_0.shape) + [1]).astype('float32')
        self.tr_pairs_1 = np.reshape(tr_pairs_1, list(tr_pairs_1.shape) + [1]).astype('float32')
        self.val_pairs_0 = np.reshape(val_pairs_0, list(val_pairs_0.shape) + [1]).astype('float32')
        self.val_pairs_1 = np.reshape(val_pairs_1, list(val_pairs_1.shape) + [1]).astype('float32')

        self.batch_size = batch_size
        self.samples_per_train = self.tr_pairs_0.shape[0] // self.batch_size
        self.samples_per_val = self.val_pairs_0.shape[0] // self.batch_size

        self.cur_train_index = 0
        self.cur_val_index = 0

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
            yield ([self.tr_pairs_0[self.cur_train_index:self.cur_train_index + self.batch_size],
                    self.tr_pairs_1[self.cur_train_index:self.cur_train_index + self.batch_size]
                    ],
                   self.tr_y[self.cur_train_index:self.cur_train_index + self.batch_size]
                   )

    def next_val(self):
        while 1:
            self.cur_val_index += self.batch_size
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index = 0
            yield ([self.val_pairs_0[self.cur_val_index:self.cur_val_index + self.batch_size],
                    self.val_pairs_1[self.cur_val_index:self.cur_val_index + self.batch_size]
                    ],
                   self.val_y[self.cur_val_index:self.cur_val_index + self.batch_size]
                   )


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    # seq = Sequential()
    # seq.add(Dense(1028, input_shape=(input_dim,), activation='relu'))
    # seq.add(Dropout(0.1))
    # seq.add(Dense(1028, activation='relu'))
    # seq.add(Dropout(0.1))
    # seq.add(Dense(1028, activation='relu'))
    # return seq
    input = Input(shape=input_shape, name="base_input")
    # x = Flatten(name="flatten_input")(input)
    # x = Dense(128, activation='relu', name="first_base_dense")(x)
    # x = Dropout(0.3, name="first_dropout")(x)
    # x = Dense(128, activation='relu', name="second_base_dense")(x)
    # x = Dropout(0.3, name="second_dropout")(x)
    # x = Dense(128, activation='relu', name="third_base_dense")(x)
    x = Convolution2D(16, (8, 8), strides=(1, 1), activation="relu", input_shape=input_shape)(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Convolution2D(32, (4, 4), strides=(1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    # x = Dense(10, activation='softmax')(x)

    # Returning a Model, with input and outputs, not just a group of layers.
    return Model(inputs=input, outputs=x)


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def train_model():
    input_dim = 5040
    input_dim_a = 40
    input_dim_b = 126
    epochs = 100
    batch_size = 128

    datagen = DataGenerator(batch_size)

    steps_per_epoch = TRAIN_NUM/batch_size
    val_steps = VAL_NUM / batch_size
    emb_size = 128



    # network definition
    # base_network = create_base_network(input_dim)
    base_network = create_base_network((input_dim_a, input_dim_b, 1))

    # input_a = Input(shape=(input_dim,), name='left_input')
    # input_b = Input(shape=(input_dim,), name='right_input')
    input_a = Input(shape=(input_dim_a, input_dim_b, 1), name='left_input')
    input_b = Input(shape=(input_dim_a, input_dim_b, 1), name='right_input')

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, name='output_layer', output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b])
    y = Dense(256, activation='sigmoid')(distance)  # relu or sigmoid
    model = Model(inputs=[input_a, input_b], outputs=y)
    print(model.summary())

    name = "snn-article-example-run"
    logdir = os.path.join("reports/", name)

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    #
    # # Callbacks:
    # # Create the TensorBoard callback
    # tensorboard = TensorBoard(
    #     log_dir=logdir,
    #     histogram_freq=0,
    #     batch_size=batch_size,
    #     write_graph=True,
    #     write_grads=True,
    #     write_images=True,
    #     update_freq='epoch',
    #     profile_batch=0
    # )
    # # Training logger
    # csv_log = os.path.join(logdir, 'training.csv')
    # csv_logger = CSVLogger(csv_log, separator=',', append=True)
    # Only save the best model weights based on the val_loss
    checkpoint = ModelCheckpoint(os.path.join(logdir, 'snn_model-{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss',
                                    save_best_only=True, verbose=1, save_weights_only=True, mode='auto')

    # # Save the embedding mode weights based on the main model's val loss
    # # This is needed to reecreate the emebedding model should we wish to visualise
    # # the latent space at the saved epoch
    # class SaveEmbeddingModelWeights(Callback):
    #     def __init__(self, filepath, monitor='val_loss', verbose=1):
    #         super(Callback, self).__init__()
    #         self.monitor = monitor
    #         self.verbose = verbose
    #         self.best = np.Inf
    #         self.filepath = filepath
    #
    #     def on_epoch_end(self, epoch, logs=None):
    #         if logs is None:
    #             logs = {}
    #         current = logs.get(self.monitor)
    #         if current is None:
    #             warnings.warn("SaveEmbeddingModelWeights requires %s available!" % self.monitor, RuntimeWarning)
    #
    #         if current < self.best:
    #             filepath = self.filepath.format(epoch=epoch + 1, **logs)
    #             # if self.verbose == 1:
    #             # print("Saving embedding model weights at %s" % filepath)
    #             base_network.save_weights(filepath, overwrite=True)
    #             self.best = current
    #
    # # Save the embedding model weights if you save a new snn best model based on the model checkpoint above
    # emb_weight_saver = SaveEmbeddingModelWeights(os.path.join(logdir, 'emb_model-{epoch:02d}.h5'))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=2)
    callbacks = [checkpoint, early_stopping]

    # Save model configs to JSON
    model_json = model.to_json()
    with open(os.path.join(logdir, "siamese_config.json"), "w") as json_file:
        json_file.write(model_json)
        json_file.close()
    model_json = base_network.to_json()
    with open(os.path.join(logdir, "base_network_config.json"), "w") as json_file:
        json_file.write(model_json)
        json_file.close()

    hyperparams = {'batch_size': batch_size,
                   'epochs': epochs,
                   'steps_per_epoch': steps_per_epoch,
                   'val_steps': val_steps,
                   'emb_size': emb_size
                   }
    with open(os.path.join(logdir, "hyperparams.json"), "w") as json_file:
        json.dump(hyperparams, json_file)

    print("Starting training process!")
    print("-------------------------------------")

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
    print(model.summary())

    siamese_history = model.fit(x=datagen.next_train(), steps_per_epoch=datagen.samples_per_train,
                                epochs=epochs, verbose=2, callbacks=callbacks,
                                validation_data=datagen.next_val(), validation_steps=datagen.samples_per_val)
    # model.load_weights("siamese_checkpoint.h5")
    print("-------------------------------------")
    print("Training complete.")

    # list all data in history
    print(siamese_history.history.keys())
    # summarize history for accuracy
    plt.plot(siamese_history.history['accuracy'])
    plt.plot(siamese_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(siamese_history.history['loss'])
    plt.plot(siamese_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # def json_to_dict(json_src):
    #     with open(json_src, 'r') as j:
    #         return json.loads(j.read())
    #
    # ## Load in best trained SNN and emb model
    #
    # # The best performing model weights has the higher epoch number due to only saving the best weights
    # highest_epoch = 0
    # dir_list = os.listdir(logdir)
    #
    # for file in dir_list:
    #     if file.endswith(".h5"):
    #         epoch_num = int(file.split("-")[1].split(".h5")[0])
    #         if epoch_num > highest_epoch:
    #             highest_epoch = epoch_num
    #
    # # Find the embedding and SNN weights src for the highest_epoch (best) model
    # for file in dir_list:
    #     # Zfill ensure a leading 0 on number < 10
    #     if ("-" + str(highest_epoch).zfill(2)) in file:
    #         if file.startswith("emb"):
    #             embedding_weights_src = os.path.join(logdir, file)
    #         elif file.startswith("snn"):
    #             snn_weights_src = os.path.join(logdir, file)
    #
    # hyperparams = os.path.join(logdir, "hyperparams.json")
    # snn_config = os.path.join(logdir, "siamese_config.json")
    # emb_config = os.path.join(logdir, "embedding_config.json")
    #
    # snn_config = json_to_dict(snn_config)
    # emb_config = json_to_dict(emb_config)
    #
    # # json.dumps to make the dict a string, as required by model_from_json
    # loaded_snn_model = model_from_json(json.dumps(snn_config))
    # loaded_snn_model.load_weights(snn_weights_src)
    #
    # loaded_emb_model = model_from_json(json.dumps(emb_config))
    # loaded_emb_model.load_weights(embedding_weights_src)
    #
    #
    # # Store visualisations of the embeddings using PCA for display next to "after training" for comparisons
    # embeddings_after_train = loaded_emb_model.predict(x_test[:num_vis, :])
    # pca = PCA(n_components=2)
    # decomposed_embeddings_after = pca.fit_transform(embeddings_after_train)
    # evaluate(loaded_emb_model, highest_epoch)