# source :
# http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import pickle
import sys
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import h5py
import click
import lassen

use_tensorflow_debugger = False
if use_tensorflow_debugger:
    import tensorflow as tf
    sess = tf.Session()
    from tensorflow.python import debug as tf_debug
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    from keras import backend as K
    K.set_session(sess)

CACHE_FILENAME = 'cache/wonderland_preprocess.pickle'

@click.group()
def cli():
    pass

@cli.command()
def generate_text():
    with open(CACHE_FILENAME, 'rb') as input:
        cached_data = pickle.load(input)
        char_to_int = cached_data['char_to_int']
        int_to_char = cached_data['int_to_char']
        X = cached_data['X']
        y = cached_data['y']
        print("Read cache file %s." % input.name)

    # seq_length = 100
    # dataX, dataY = [], []
    # for i in range(len(raw_text) - seq_length):
    #     seq_in = raw_text[i:i + seq_length]
    #     seq_out = raw_text[i + seq_length]
    #     dataX.append([char_to_int[char] for char in seq_in])
    #     dataY.append(char_to_int[seq_out])

    start = 100000 # np.random.randint(0, len(X)-1)
    print("Using pattern %s of %s." % (start, len(X)))
    pattern = [int(x) for x in X[start] * len(char_to_int)]
    lstm_length = 4

    timesteps = 100
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    # model.add(LSTM(lstm_length, input_shape=(timesteps, 1), stateful=True, batch_input_shape=(1,timesteps,1)))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.load_weights('weights/weights-improvement-19-2.0261.hdf5')

    filename = 'weights/weights-improvement-19-2.0261.hdf5'
    f = h5py.File(filename)
    dense_kernel = f['model_weights']['dense_1']['dense_1']['kernel:0']
    dense_bias = f['model_weights']['dense_1']['dense_1']['bias:0']

    # lstm_kernel = f['model_weights']['lstm_1']['lstm_1']['kernel:0'][()]
    # lstm_recurrent_kernel = f['model_weights']['lstm_1']['lstm_1']['recurrent_kernel:0'][()]
    # lstm_bias = f['model_weights']['lstm_1']['lstm_1']['bias:0'][()]
    # model.layers[0].set_weights([lstm_kernel, lstm_recurrent_kernel, lstm_bias])

    # print("recurrent_kernel", lstm_recurrent_kernel)
    # print("kernel", lstm_kernel)
    #


    network = [
        lassen.LSTMLayer(1, 256),
        lassen.DenseLayer(256, y.shape[1]),
        lassen.SoftmaxLayer(256)
    ]
    # print('y.shape[1]', y.shape[1])
    # print('len(int_to_char)', len(int_to_char))
    # print("Dense shape", network[1].weights.shape)
    # print("Keras shape", dense_kernel.shape)
    # print("lstm kernel shape", lstm_kernel.shape)
    # print("lstm recurrent kernel shape", lstm_recurrent_kernel.shape)
    # print("lstm bias shape", lstm_bias.shape)
#    network[0].set_keras_weights([lstm_kernel, lstm_recurrent_kernel, lstm_bias])
    network[0].set_keras_weights(model.layers[0].get_weights())
    network[1].weights[:] = dense_kernel
    network[1].biases[:] = dense_bias

    # generate characters
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(int_to_char))

        # lstm_layers = model.layers[0]
        # intermediate_layer_model = Model(inputs=model.input,
        #     outputs = lstm_layers.output)
        # output = intermediate_layer_model.predict(x)

        # import tensorflow as tf
        # with tf.Session().as_default():
        #     print("Lstm_layers", lstm_layers)
        #     print("Lstm_layers dir", dir(lstm_layers))
        #     print("Lstm_layers adrien_test_i", lstm_layers.adrien_test_i)
        #     print("Lstm_layers adrien_test_i eval", lstm_layers.adrien_test_i.eval())
        # exit()

        # # lukas debug code
        # print(tf.get_default_graph().get_operations())
        # #print(tf.get_default_graph().get_tensors())
        # tensor = tf.get_default_graph().get_tensor_by_name("lstm_1/mul_9:0")
        # with sess.as_default():
        #     print(tensor.eval())
        # tf.Print(tensor, [tensor])
        # print("keras intermediate output", output)
        # #print("prediction", prediction)
        #
        # #print(x[0])

        # lassen_intermediate = network[0].forward(x[0,:])
        # print("lassen intermediate output", lassen_intermediate)

        keras_prediction = model.predict(x)
        lassen_prediction = lassen.forward(network, x[0,:])
        # print('lassen prediction', lassen_prediction)
        # print('lassen prediction', np.exp(lassen_prediction))
        # exit()

        #print(prediction[0])
        #index = np.random.choice(range(len(prediction[0])), p = prediction[0] ** 2 / sum(prediction[0] ** 2))
        index = np.argmax(lassen_prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        sys.stdout.flush()
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")

# tou thin i sesul tauh tou th toe the tores.'
#
# 'tou dre't toin tou,  saed the darerpillar.
#
# 'the mant tase thu mote ' said the marth hare.
#
# 'toe taan toeh to tel that you toun the thit!' shi gact sa the gury, and the part of the^C

@cli.command()
def preprocess_text():
    # load ascii text and covert to lowercase
    filename = "wonderland.txt"
    raw_text = open(filename).read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i,c) for (c,i) in char_to_int.items())

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX, dataY = [], []
    for i in range(len(raw_text) - seq_length):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(len(chars))
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    print(len(char_to_int))
    print(len(int_to_char))
    print(X.shape)
    print(y.shape)

    # save the preprocessed text
    cached_data = {
        'char_to_int': char_to_int,
        'int_to_char': int_to_char,
        'X': X,
        'y': y
    }
    with open(CACHE_FILENAME, 'wb') as output:
        pickle.dump(cached_data, output)
        print("Wrote cache to %s." % output.name)

@cli.command()
def learn_model():
    with open(CACHE_FILENAME, 'rb') as input:
        cached_data = pickle.load(input)
        char_to_int = cached_data['char_to_int']
        int_to_char = cached_data['int_to_char']
        X = cached_data['X']
        y = cached_data['y']
        print("Read cache file %s." % input.name)

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

if __name__ == "__main__":
    cli()
