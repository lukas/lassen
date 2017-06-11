import keras
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model, load_model
from tensorflow.python import pywrap_tensorflow
import data

import keras.regularizers
import numpy as np

import weights


def display_images(images, labels, row_count, col_count):
    for label, image in zip(labels, images):
        for c in range(col_count):
            for r in range(row_count):
                if image[c*row_count + r] == 0:
                    print(" ",end='')
                else:
                    print("*",end='')
            print("")
        print(label)

def build_two_layer_model():
    images, one_hot_labels = data.load_train_mnist()
    test_images, one_hot_test_labels = data.load_test_mnist()
    #display_images(images, labels, 28, 28)
    num_classes= 10
    num_images = images.shape[0]
    num_pixels = images.shape[1]

    num_test_images = test_images.shape[0]

    images = images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    model=Sequential()
    model.add(Dense(100, activation='relu', input_shape=(num_pixels,), name="main" ))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer="sgd", verbose=2, loss= 'categorical_crossentropy', metrics=['accuracy'])

    model.fit(images, one_hot_labels, validation_data=(test_images, one_hot_test_labels), epochs=3)
    model.save('two_layer.h5')

def build_perceptron():
    print("Building Perceptron")
    images, one_hot_labels = data.load_train_mnist()
    test_images, one_hot_test_labels = data.load_test_mnist()

    num_classes= 10
    num_images = images.shape[0]
    images = images.reshape((num_images,28,28,1))


    num_pixels = images.shape[1]
    model=Sequential()
    model.add(Flatten(input_shape=(num_pixels,num_pixels,1)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer="sgd", verbose=2, loss= 'categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, one_hot_labels, epochs=1)

    model.save("perceptron.h5")
    print("Built Perceptron")

def build_small_conv_model():
    images, one_hot_labels = data.load_train_mnist()
    test_images, one_hot_test_labels = data.load_test_mnist()

    num_classes= 10
    num_images = images.shape[0]
    images = images.reshape((num_images,28,28,1))

    num_pixels = images.shape[1]

    num_test_images = test_images.shape[0]
    test_images = test_images.reshape((num_test_images,28,28,1))

    model=Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=(num_pixels,num_pixels,1), padding="same", activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(num_classes, activation='softmax', name="final"))

    print(model.summary())
    print(test_images.shape)

    model.compile(optimizer="sgd", verbose=2, loss= 'categorical_crossentropy', metrics=['accuracy'])
    #model.save("small_conv_improved.h5")

    model.fit(images, one_hot_labels, validation_data=(test_images, one_hot_test_labels), epochs=3)
    model.save("small_conv_improved.h5")

def build_identity_conv_model():
    msimple = Sequential()
    msimple.add(Conv2D(1, (1, 1), input_shape=(28,28,1), padding="same"))
    w = msimple.layers[0].get_weights()
    w[0][0,0,0,0]=1.0
    msimple.layers[0].set_weights(w)

def test_model(model_file):
    test_images, one_hot_test_labels = data.load_test_mnist()
    num_test_images = test_images.shape[0]

    test_images = test_images.reshape((num_test_images,28,28,1))

    model = load_model(model_file)
    print(model.summary())

    num_layers = 7
    for layer in range(num_layers):
        if model.layers[layer].get_weights():
            print("Weights Shape", model.layers[layer].get_weights()[0].shape)
            if (layer == 5):
                print("Weights: ", model.layers[layer].get_weights()[0])
        intermediate_layer_model = Model(inputs=model.input,
            outputs = model.layers[layer].output)
        guesses = intermediate_layer_model.predict(test_images[:1])
        #if layer == 5:
        print("KERAS Layer %s SUM: %s SHAPE: %s" % \
            (layer, np.sum(guesses), guesses.shape))
        #if layer == 3:
        #    print("Layer %i Output %s" % (layer, guesses))

if __name__ == "__main__":
#    build_small_conv_model()
    test_model("small_conv_improved.h5")
