import json
import warnings
import matplotlib.pyplot as plt
from keras.api._v2.keras.models import Model, load_model
from keras.api._v2.keras.layers import Dense, Dropout, Input, Flatten, Convolution2D, MaxPooling2D, Layer
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras import backend as K
from keras.api._v2.keras.callbacks import ModelCheckpoint, CSVLogger, Callback, ReduceLROnPlateau
from keras.api._v2.keras.metrics import BinaryAccuracy
from keras.api._v2.keras.losses import BinaryCrossentropy
from keras.api._v2.keras.utils import plot_model
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

warnings.filterwarnings('ignore')

LOG = get_logger('SiameseNet')


class SiameseNet:
    def __init__(self, shape):
        self.config = my_config.get_config()
        self.epochs = self.config['epochs']
        self.batch_size = self.config['batch_size']
        # self.input_dim_a = self.config['input_dim_a']
        # self.input_dim_b = self.config['input_dim_b']
        self.input_dim_a = shape[0]
        self.input_dim_b = shape[1]
        self.input_channels = 1
        self.datagen_val = None
        self.datagen_train = None
        self.identical_subnetwork = None
        self.model = None
        self.callbacks = []
        self.history = None
        self.training_time = None
        self.trainable_count = None
        self.logdir = os.path.join(self.config['log']['dir'], self.config['log']['name'])
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
        distance = DistanceLayer(k=outputs_shape, name='contrast_function')([processed_a, processed_b])
        # dense = Dense(units=outputs_shape//2, activation='relu')(distance)
        output = Dense(1, activation='sigmoid', name='output_layer')(distance)  # relu or sigmoid

        self.model = Model(inputs=[input_a, input_b], outputs=output)
        print(self.model.summary())
        plot_model(self.model, to_file=os.path.join(self.logdir, "model.png"), show_shapes=True, show_layer_names=True,
                   show_layer_activations=True)
        plot_model(self.model, to_file=os.path.join(self.logdir, "model_expand_nested.png"), show_shapes=True,
                   show_layer_names=True, expand_nested=True,
                   show_layer_activations=True)
        self.trainable_count = count_params(self.model.trainable_weights)

    def _identical_subnetwork(self):
        config = my_config.get_config()
        inputs = Input(shape=(self.input_dim_a, self.input_dim_b, self.input_channels), name='base_input')

        cnn = config['layers']['cnn']
        denses = config['layers']['dns']
        flatten = config['layers']['flt']

        # first layer of convolution2D
        model = Convolution2D(filters=cnn['conv']['filters'][0], kernel_size=cnn['conv']['kernel'][0],
                              strides=cnn['conv']['stride'][0], activation=cnn['conv']['activation'][0])(inputs)
        if cnn['dropout'][0] is not None:
            model = Dropout(cnn['dropout'][0])(model)
        if cnn['pool']['size'][0] is not None:
            model = MaxPooling2D(pool_size=cnn['pool']['size'][0], strides=cnn['pool']['stride'][0])(model)

        # additional layers
        for idx in range(cnn['num_of_layers'])[1:]:
            if cnn['dropout'][idx] is not None:
                model = Convolution2D(filters=cnn['conv']['filters'][idx], kernel_size=cnn['conv']['kernel'][idx],
                                      strides=cnn['conv']['stride'][idx], activation=cnn['conv']['activation'][idx])(
                    model)
            if cnn['dropout'][idx] is not None:
                model = Dropout(cnn['dropout'][idx])(model)
            if cnn['pool']['size'][idx] is not None:
                model = MaxPooling2D(pool_size=cnn['pool']['size'][idx], strides=cnn['pool']['stride'][idx])(model)

        # flatten
        if flatten is True:
            model = Flatten()(model)

        # dense layers
        for idx in range(denses['num_of_layers']):
            model = Dense(units=denses['units'][idx], activation=denses['activation'][idx])(model)
            if denses['dropout'][idx] is not None:
                model = Dropout(denses['dropout'][idx])(model)

        return Model(inputs, model)

    def train(self):
        LOG.info('Loading train dataset to DataGenerator...')
        self.datagen_train = SiameseGenerator('train')
        LOG.info('Loading validation dataset to DataGenerator...')
        self.datagen_val = SiameseGenerator('val', shuffle=False)

        # Callbacks:
        LOG.info("Setting callback functions...")
        # Only save the best model weights based on the val_loss
        checkpoint = ModelCheckpoint(os.path.join(self.logdir, 'snn_model-{epoch:02d}-{val_loss:.2f}.h5'),
                                     monitor='val_loss', save_best_only=True, verbose=1, mode='min')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1,
                                                          min_delta=1e-3)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=3, min_lr=0.001, verbose=1)
        # Training logger
        csv_logger = CSVLogger(os.path.join(self.logdir, 'training.csv'), separator=',', append=True)
        # Save the embedding model weights if you save a new snn best model based on the model checkpoint above
        emb_weight_saver = SaveEmbeddingModelWeights(filepath=os.path.join(self.logdir, 'emb_model-{epoch:02d}.h5'),
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
            optimizer=Adam(lr=self.config['learning_rate']),
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
        plt.savefig(os.path.join(self.logdir, 'loss.png'))

    def evaluate(self):
        # Evaluate the model
        LOG.info("Evaluating model...")
        datagen_test_eval = SiameseGenerator("test", shuffle=False)

        scores = self.model.evaluate(x=datagen_test_eval, steps=len(datagen_test_eval), verbose=1)  # steps=16

        print(dict(zip(self.model.metrics_names, scores)))

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
        data = read_csv(f'{self.config["pairs_root"]}{self.config["pairs_name"]["test"]}', delimiter=';')
        true_y = data["label"].to_numpy().astype('float32')
        print(classification_report(true_y, pred_y))

        Accuracy = accuracy_score(true_y, pred_y)
        Precision = precision_score(true_y, pred_y)
        Sensitivity_recall = recall_score(true_y, pred_y)
        Specificity = recall_score(true_y, pred_y, pos_label=0)
        F1_score = f1_score(true_y, pred_y)
        tn, fp, fn, tp = confusion_matrix(true_y, pred_y).ravel()
        print(f"TYPE SCORE[0]: {type(scores[0])}")
        print(f"TYPE TN: {type(tn)}")
        metrics = {"Training time (hour:minute:second)": self.training_time, "Trainable params": self.trainable_count,
                   "Test loss": scores[0], "Test accuracy": scores[1],
                   "Accuracy": Accuracy, "Precision": Precision,
                   "Sensitivity_recall": Sensitivity_recall, "Specificity": Specificity, "F1_score": F1_score,
                   "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        with open(os.path.join(self.logdir, 'metrics.json'), 'w') as json_file:
            json.dump(metrics, json_file)
            json_file.close()

        conf_matrix = confusion_matrix(true_y, pred_y, normalize='all')
        # TODO maybe we should rename True and False in display_labels.
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[False, True])
        plt.ioff()
        cm_display.plot()
        plt.savefig(os.path.join(self.logdir, 'confusion_matrix.png'))

    def restore_model(self, file: str):
        self.model = load_model(file, custom_objects={'contrast_function': DistanceLayer})
        print(self.model.summary())
        plot_model(self.model, to_file=os.path.join(self.logdir, "model_restored.png"), show_shapes=True,
                   show_layer_names=True,
                   show_layer_activations=True, expand_nested=True)


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
