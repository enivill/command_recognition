import json
import warnings
import matplotlib.pyplot as plt
from keras.api._v2.keras.models import Model, load_model
from keras.api._v2.keras.layers import Dense, Dropout, Input, Flatten, Convolution2D, MaxPooling2D, Layer, \
    BatchNormalization, Activation
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras import backend as K
from keras.api._v2.keras.callbacks import ModelCheckpoint, CSVLogger, Callback, ReduceLROnPlateau
from keras.api._v2.keras.metrics import BinaryAccuracy
from keras.api._v2.keras.losses import BinaryCrossentropy
from keras.api._v2.keras.utils import plot_model
from keras.api._v2.keras.activations import gelu, relu
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, \
    ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params
import os
import numpy as np
from src.models.generators import SiameseGenerator
from src.utils import config as my_config
from src.utils.logger import get_logger
import yaml
from pandas import read_csv
import time
import csv
from src.features.build_features import feature_extraction

warnings.filterwarnings('ignore')

LOG = get_logger('SiameseNet')


class SiameseNet:
    def __init__(self, restore_model: bool = False):
        self.config = my_config.get_config()
        self.restore_model = restore_model
        if not self.restore_model:
            self.epochs = self.config['train']['epochs']
            self.input_dim_a, self.input_dim_b = self._get_feature_shape(self.config)
            self.input_channels = 1
            self.datagen_val = None
            self.datagen_train = None
            self.identical_subnetwork = None
            self.callbacks = []
            self.history = None
            self.training_time = None
            self.trainable_count = None
        self.model = None
        self.batch_size = self.config['train']['batch_size']
        self.logdir = os.path.join(self.config['train']['log']['dir'], self.config['train']['log']['name'])
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)

    def build(self):
        """Builds the siamese_network"""
        LOG.info('Building siamese network...')
        self.identical_subnetwork = self._identical_subnetwork()
        print(self.identical_subnetwork.summary())
        plot_model(self.identical_subnetwork, to_file=os.path.join(self.logdir, "subnetwork.png"), show_shapes=True,
                   show_layer_names=True, show_layer_activations=True)

        input_a = Input(shape=(self.input_dim_a, self.input_dim_b, self.input_channels), name='left_input')
        input_b = Input(shape=(self.input_dim_a, self.input_dim_b, self.input_channels), name='right_input')

        # because we re-use the same instance `identical_subnetwork`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = self.identical_subnetwork(input_a)
        processed_b = self.identical_subnetwork(input_b)
        outputs_shape = self.config["layers"]["dns"]["units"][-1]  # the unit of the last dense layer
        model = DistanceLayer(k=outputs_shape, name='contrast_function')([processed_a, processed_b])
        denses_after_distance = self.config['layers']['after_distance']['dns']

        # dense layers
        for idx in range(len(denses_after_distance['units'])):
            model = Dense(units=denses_after_distance['units'][idx],
                          activation=denses_after_distance['activation'][idx])(model)
            if denses_after_distance['batchnorm'][idx] is True:
                model = BatchNormalization()(model)
            if denses_after_distance['dropout'][idx] is not None:
                model = Dropout(denses_after_distance['dropout'][idx])(model)

        output = Dense(1, activation='sigmoid', name='output_layer')(model)  # relu or sigmoid

        self.model = Model(inputs=[input_a, input_b], outputs=output)
        print(self.model.summary())
        plot_model(self.model, to_file=os.path.join(self.logdir, "model.png"), show_shapes=True, show_layer_names=True,
                   show_layer_activations=True)
        plot_model(self.model, to_file=os.path.join(self.logdir, "model_expand_nested.png"), show_shapes=True,
                   show_layer_names=True, expand_nested=True, show_layer_activations=True)
        plot_model(self.model, to_file=os.path.join(self.logdir, "model_simple.png"), show_shapes=False,
                   show_layer_names=False, expand_nested=True, show_layer_activations=False)
        self.trainable_count = count_params(self.model.trainable_weights)

    def _identical_subnetwork(self):
        layers_config = my_config.get_config()['layers']
        inputs = Input(shape=(self.input_dim_a, self.input_dim_b, self.input_channels), name='base_input')

        cnn = layers_config['cnn']
        denses = layers_config['dns']
        model = inputs
        # # first layer of convolution2D, there should always be at least one convolutional layer
        # model = Convolution2D(filters=cnn['conv']['filters'][0], kernel_size=cnn['conv']['kernel'][0],
        #                       strides=cnn['conv']['stride'][0], activation=cnn['conv']['activation'][0], padding='same')(inputs)
        # if cnn['pool']['size'][0] is not None:
        #     model = MaxPooling2D(pool_size=cnn['pool']['size'][0], strides=cnn['pool']['stride'][0], padding='same')(model)
        # if cnn['batchnorm'][0] is True:
        #     model = BatchNormalization()(model)
        # if cnn['dropout'][0] is not None:
        #     model = Dropout(cnn['dropout'][0])(model)

        # additional layers
        for idx in range(len(cnn['conv']['filters'])):
            # conv layer
            model = Convolution2D(filters=cnn['conv']['filters'][idx],
                                  kernel_size=cnn['conv']['kernel'][idx],
                                  strides=cnn['conv']['stride'][idx],
                                  # activation=cnn['conv']['activation'],
                                  padding='same')(model)

            # Activation function
            model = Activation(relu)(model)
            # pooling
            if cnn['pool']['size'][idx] is not None:
                model = MaxPooling2D(pool_size=cnn['pool']['size'][idx], strides=cnn['pool']['stride'][idx],
                                     padding='same')(model)
            # batch normalization
            if cnn['batchnorm'][idx] is True:
                model = BatchNormalization()(model)
            # dropout
            if cnn['dropout'][idx] is not None:
                model = Dropout(cnn['dropout'][idx])(model)

        # flatten
        model = Flatten()(model)

        # dense layers
        for idx in range(len(denses['units'])):
            model = Dense(units=denses['units'][idx], activation=denses['activation'])(model)
            if denses['batchnorm'][idx] is True:
                model = BatchNormalization()(model)
            if denses['dropout'][idx] is not None:
                model = Dropout(denses['dropout'][idx])(model)

        return Model(inputs, model)

    def train(self):
        LOG.info('Loading train dataset to DataGenerator...')
        self.datagen_train = SiameseGenerator('train', shuffle=True)
        LOG.info('Loading validation dataset to DataGenerator...')
        self.datagen_val = SiameseGenerator('val', shuffle=False)

        # Callbacks:
        LOG.info("Setting callback functions...")
        # Only save the best model weights based on the val_loss
        checkpoint = ModelCheckpoint(os.path.join(self.logdir, 'snn_model_best.h5'),
                                     monitor='val_loss', save_best_only=True, verbose=1, mode='min')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                          patience=self.config['train']['early_stopping_patience'],
                                                          verbose=1,
                                                          min_delta=self.config['train']['early_stopping_min_delta'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=self.config['train']['reduce_lr_patience'], min_lr=1e-5, verbose=1)
        # Training logger
        csv_logger = CSVLogger(os.path.join(self.logdir, 'training.csv'), separator=',', append=True)
        # Save the embedding model weights if you save a new snn best model based on the model checkpoint above
        emb_weight_saver = SaveEmbeddingModelWeights(filepath=os.path.join(self.logdir, 'emb_model_best.h5'),
                                                     identical_subnetwork=self.identical_subnetwork)
        self.callbacks = [checkpoint, early_stopping, csv_logger, emb_weight_saver, reduce_lr]

        # Save model configs to JSON
        LOG.info("Saving model configs to JSON...")
        with open(os.path.join(self.logdir, 'siamese_config.json'), 'w') as json_file:
            json_file.write(self.model.to_json())
            json_file.close()
        with open(os.path.join(self.logdir, 'identical_subnetwork_config.json'), 'w') as json_file:
            json_file.write(self.identical_subnetwork.to_json())
            json_file.close()

        # save config
        LOG.info("Saving project configs to JSON...")
        with open(os.path.join(self.logdir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)

        LOG.info("Starting training process...")

        self.model.compile(
            loss=BinaryCrossentropy(from_logits=False),
            optimizer=Adam(lr=self.config['train']['learning_rate']),
            metrics=[BinaryAccuracy()]
        )
        start = time.time()
        self.history = self.model.fit(x=self.datagen_train,
                                      steps_per_epoch=len(self.datagen_train),
                                      epochs=self.epochs,
                                      verbose=1,
                                      callbacks=self.callbacks,
                                      validation_data=self.datagen_val,
                                      validation_steps=len(self.datagen_val))
        stop = time.time()
        self.training_time = time.strftime("%H:%M:%S", time.gmtime(stop - start))
        LOG.info(f"Training time: {self.training_time}")
        # model.load_weights('siamese_checkpoint.h5')

        LOG.info("Training complete.")

        # serialize the model to disk
        LOG.info("Saving siamese model...")
        self.model.save(os.path.join(self.logdir, "model.h5"))
        self._write_summary()

    def _write_summary(self):

        # list all data in history
        LOG.info(f"History keys: {self.history.history.keys()}")
        # plot the training history
        LOG.info("Summarizing history for accuracy and loss...")
        plt.ioff()
        # summarize history for accuracy
        fig = plt.figure()
        plt.plot(self.history.history['binary_accuracy'])
        plt.plot(self.history.history['val_binary_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.annotate('%0.2f' % self.history.history['binary_accuracy'][-1],
                     xy=(1, self.history.history['binary_accuracy'][-1]), xytext=(8, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points', color='C0')
        plt.annotate('%0.2f' % self.history.history['val_binary_accuracy'][-1],
                     xy=(1, self.history.history['val_binary_accuracy'][-1]), xytext=(8, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points', color='C1')
        plt.savefig(os.path.join(self.logdir, 'accuracy.png'))
        plt.close(fig)
        # summarize history for loss
        fig = plt.figure()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.annotate('%0.2f' % self.history.history['loss'][-1],
                     xy=(1, self.history.history['loss'][-1]), xytext=(8, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points', color='C0')
        plt.annotate('%0.2f' % self.history.history['val_loss'][-1],
                     xy=(1, self.history.history['val_loss'][-1]), xytext=(8, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points', color='C1')
        plt.savefig(os.path.join(self.logdir, 'loss.png'))

    def evaluate(self):
        # Evaluate the model
        LOG.info("Evaluating model...")
        datagen_test_eval = SiameseGenerator("test", shuffle=False)

        scores = self.model.evaluate(x=datagen_test_eval, steps=len(datagen_test_eval), verbose=1)  # steps=16

        # print(dict(zip(self.model.metrics_names, scores)))
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        LOG.info("Predict model...")
        datagen_test_pred = SiameseGenerator("test", shuffle=False, to_fit=False)
        # predict return probabilities by sigmoid
        y_prob = self.model.predict(x=datagen_test_pred, steps=len(datagen_test_pred), verbose=1)
        # y_classes = y_prob.argmax(axis=-1)
        pred_y = [1 * (x[0] >= 0.5) for x in y_prob]
        data = read_csv(f'{self.config["paths"]["pairs_root"]}{self.config["paths"]["pairs_name"]["test"]}',
                        delimiter=';')
        true_y = data["label"].to_numpy().astype('float32')
        print(classification_report(true_y, pred_y))

        Accuracy = accuracy_score(true_y, pred_y)
        Precision = precision_score(true_y, pred_y)
        Sensitivity_recall = recall_score(true_y, pred_y)
        Specificity = recall_score(true_y, pred_y, pos_label=0)
        F1_score = f1_score(true_y, pred_y)
        tn, fp, fn, tp = confusion_matrix(true_y, pred_y).ravel()
        one_percent = (tn + fp + fn + tp) / 100

        if not self.restore_model:
            training_settings = {'model_path': self.config['train']['log']['name'],
                                 'word_per_class': self.config['make_pairs']['word_per_class_train'],
                                 'sr': self.config['feature_extraction']['sample_rate'],
                                 'wl': self.config['feature_extraction']['window_length_seconds'],
                                 'hl': self.config['feature_extraction']['hop_length_seconds'],
                                 'mels': self.config['feature_extraction']['n_mels'],
                                 'f_min': self.config['feature_extraction']['f_min'],
                                 'f_max': self.config['feature_extraction']['f_max'],
                                 'cnn_filters': self.config['layers']['cnn']['conv']['filters'],
                                 'cnn_kernel': self.config['layers']['cnn']['conv']['kernel'],
                                 'cnn_stride': self.config['layers']['cnn']['conv']['stride'],
                                 'cnn_activation': self.config['layers']['cnn']['conv']['activation'],
                                 'cnn_dropout': self.config['layers']['cnn']['dropout'],
                                 'cnn_batch_norm': self.config['layers']['cnn']['batchnorm'],
                                 'cnn_pool_size': self.config['layers']['cnn']['pool']['size'],
                                 'cnn_pool_stride': self.config['layers']['cnn']['pool']['stride'],
                                 'dns_units': self.config['layers']['dns']['units'],
                                 'dns_activation': self.config['layers']['dns']['activation'],
                                 'dns_dropout': self.config['layers']['dns']['dropout'],
                                 'after_distance_dns_unit': self.config['layers']['after_distance']['dns']['units'],
                                 'after_distance_dns_act': self.config['layers']['after_distance']['dns']['activation'],
                                 'after_distance_dns_drop': self.config['layers']['after_distance']['dns']['dropout']}
        else:
            training_settings = {'model_path': self.config['paths']['restore_model']}
        if not self.restore_model:
            metrics = {"Time(H:M:S)": self.training_time, "Params": self.trainable_count,
                       "Test loss": scores[0],
                       "Accuracy": Accuracy, "Precision": Precision,
                       "Sensitivity_recall": Sensitivity_recall, "Specificity": Specificity, "F1": F1_score,
                       "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                       "TN %": str((tn / one_percent)), "FP %": str((fp / one_percent)),
                       "FN %": str((fn / one_percent)), "TP %": str((tp / one_percent)), "Test accuracy": scores[1]*100}
        else:
            metrics = {"Test loss": scores[0], "Test accuracy": scores[1],
                       "Accuracy": Accuracy, "Precision": Precision,
                       "Sensitivity_recall": Sensitivity_recall, "Specificity": Specificity, "F1": F1_score,
                       "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                       "TN %": str((tn / one_percent)), "FP %": str((fp / one_percent)),
                       "FN %": str((fn / one_percent)), "TP %": str((tp / one_percent))}

        # append model results to csv
        self._save_model_result_to_csv(metrics, training_settings)

        with open(os.path.join(self.logdir, 'metrics.json'), 'w') as json_file:
            json.dump(metrics, json_file)
            json_file.close()

        conf_matrix = confusion_matrix(true_y, pred_y, normalize='all')
        # TODO maybe we should rename True and False in display_labels.
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[False, True])
        plt.ioff()
        cm_display.plot()
        plt.savefig(os.path.join(self.logdir, 'confusion_matrix.png'))

    def _save_model_result_to_csv(self, metrics: dict, train_settings: dict):
        # TODO change headers, test it
        filename = os.path.join(self.config['train']['log']['dir'], self.config['paths']['results_csv'])
        file_exists = os.path.isfile(filename)
        with open(filename, 'a') as csvfile:
            headers = [*metrics.keys(), *train_settings.keys()]
            writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=';', lineterminator='\r')

            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header

            writer.writerow(metrics | train_settings)

    def restore(self):
        self.model = load_model(self.config['paths']['restore_model'],
                                custom_objects={'contrast_function': DistanceLayer})
        print(self.model.summary())
        plot_model(self.model, to_file=os.path.join(self.logdir, "model_restored.png"), show_shapes=True,
                   show_layer_names=True,
                   show_layer_activations=True, expand_nested=True)

    def _get_feature_shape(self, config):
        with open(f'{config["paths"]["pairs_root"]}{config["paths"]["pairs_name"]["train"]}', newline='') as f:
            csv_reader = csv.reader(f, delimiter=';')
            csv_headings = next(csv_reader)
            first_line = next(csv_reader)
        feature = feature_extraction(first_line[0])
        return feature.shape


@tf.keras.utils.register_keras_serializable()
class DistanceLayer(Layer):
    """
    This layer is responsible for computing the distance between the two embeddings
    """

    def __init__(self, k, name, **kwargs):
        super(DistanceLayer, self).__init__(name=name)
        self.k = k
        super(DistanceLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(DistanceLayer, self).get_config()
        config["k"] = self.k
        return config

    def call(self, inputs):
        x, y = inputs
        return K.abs(x - y)
        # return K.square(x-y)


# source: https://github.com/Trotts/Siamese-Neural-Network-MNIST-Triplet-Loss/blob/main/Siamese-Neural-Network-MNIST.ipynb
# Save the embedding mode weights based on the main model's val loss
# This is needed to reecreate the emebedding model should we wish to visualise
# the latent space at the saved epoch
class SaveEmbeddingModelWeights(Callback):
    def __init__(self, filepath, identical_subnetwork, monitor='val_loss', verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.best = np.Inf
        self.filepath = filepath
        self.identical_subnetwork = identical_subnetwork

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("SaveEmbeddingModelWeights requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.best:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            # if self.verbose == 1:
            # print("Saving embedding model weights at %s" % filepath)
            self.identical_subnetwork.save_weights(filepath, overwrite=True)
            self.best = current
