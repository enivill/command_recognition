# source:
# title: Example code for Siamese Neural Network
# autor: Yash Arora
# webpage: https://roboticswithpython.com/example-code-for-siamese-neural-network/

from keras.api._v2.keras.layers import Flatten, Dense, Dropout, Input, Lambda
from keras.api._v2.keras.models import Model
from keras.api._v2.keras import backend as K
from keras.api._v2.keras.utils import plot_model
from keras.api._v2.keras.optimizers import RMSprop
import numpy as np
import random
from keras.api._v2.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)


def create_pairs_on_set(images, labels):
    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')

    return pairs, y


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def initialize_base_network():
    input = Input(shape=(28, 28,), name="base_input")
    x = Flatten(name="flatten_input")(input)
    x = Dense(128, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(128, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    x = Dense(128, activation='relu', name="third_base_dense")(x)

    return Model(inputs=input, outputs=x)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def siamese_network():
    # load the dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(f"TYPE of train_images: {type(train_images)}")  # <class 'numpy.ndarray'>
    print(f"Shape of train_images: {train_images[0].shape}")  # (28, 28)
    # prepare train and test sets
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    print(f"TYPE of train_images after float32: {type(train_images)}")  # <class 'numpy.ndarray'>
    print(f"Shape of train_images after float32: {train_images[0].shape}")  # (28, 28)

    # normalize values
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    print(f"TYPE of train_images after normalization: {type(train_images)}")  # <class 'numpy.ndarray'>
    print(f"Shape of train_images after normalization: {train_images[0].shape}")  # (28, 28)

    # create pairs on train and test sets
    tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)
    ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)
    # array index
    this_pair = 8

    # show images at this index
    show_image(ts_pairs[this_pair][0])
    show_image(ts_pairs[this_pair][1])

    # print the label for this pair
    print(ts_y[this_pair])  # 1.0

    base_network = initialize_base_network()
    plot_model(base_network, show_shapes=True, show_layer_names=True, to_file='base-model.png')
    # create the left input and point to the base network
    input_a = Input(shape=(28, 28,), name="left_input")
    vect_output_a = base_network(input_a)
    print(f"vect_output_a: {vect_output_a}")  # KerasTensor(type_spec=TensorSpec(shape=(None, 128), dtype=tf.float32, name=None),
                                              # name='model/third_base_dense/Relu:0', description="created by layer 'model'")
    print(f"input_a: {input_a}")  # KerasTensor(type_spec=TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name='left_input'),
                                  # name='left_input', description="created by layer 'left_input'")

    # create the right input and point to the base network
    input_b = Input(shape=(28, 28,), name="right_input")
    vect_output_b = base_network(input_b)

    # measure the similarity of the two vector outputs
    output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)(
        [vect_output_a, vect_output_b])

    # specify the inputs and output of the model
    model = Model([input_a, input_b], output)

    # plot model graph
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')

    rms = RMSprop()
    model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, epochs=5, batch_size=128,
                        validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]], ts_y))
    loss = model.evaluate(x=[ts_pairs[:, 0], ts_pairs[:, 1]], y=ts_y)

    y_pred_train = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    train_accuracy = compute_accuracy(tr_y, y_pred_train)

    y_pred_test = model.predict([ts_pairs[:, 0], ts_pairs[:, 1]])
    test_accuracy = compute_accuracy(ts_y, y_pred_test)

    def plot_metrics(metric_name, title, ylim=5):
        plt.title(title)
        plt.ylim(0, ylim)
        plt.plot(history.history[metric_name], color='blue', label=metric_name)
        plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)

    print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))
    plot_metrics(metric_name='loss', title="Loss", ylim=1)


def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    return contrastive_loss


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)
