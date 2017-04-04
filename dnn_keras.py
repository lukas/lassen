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
    images, one_hot_labels = data.load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
    test_images, one_hot_test_labels = data.load_mnist("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte")
    #display_images(images, labels, 28, 28)
    num_classes= 10
    num_images = images.shape[0]
    num_pixels = images.shape[1]

    num_test_images = test_images.shape[0]

    images = images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    images /= 255.0
    test_images /= 255.0

    model=Sequential()
    model.add(Dense(50, activation='relu', input_shape=(num_pixels,), name="main" ))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer="sgd", verbose=2, loss= 'categorical_crossentropy', metrics=['accuracy'], batch_size=100, epochs=2)

    model.fit(images, one_hot_labels, validation_data=(test_images, one_hot_test_labels))
    model.save('two_layer.h5')

def build_small_conv_model():
    images, one_hot_labels = data.load_normalized_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
    test_images, one_hot_test_labels = data.load_normalized_mnist("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte")

    num_classes= 10
    num_images = images.shape[0]
    images = images.reshape((num_images,28,28,1))

    num_pixels = images.shape[1]

    num_test_images = test_images.shape[0]
    test_images = test_images.reshape((num_test_images,28,28,1))

    model=Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(num_pixels,num_pixels,1), padding="same"))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(num_classes, activation='softmax', name="final"))

    print(model.summary())
    print(test_images.shape)

    model.compile(optimizer="sgd", verbose=2, loss= 'categorical_crossentropy', metrics=['accuracy'])
    model.save("small_conv_improved.h5")

    model.fit(images, one_hot_labels, validation_data=(test_images, one_hot_test_labels), batch_size=100, epochs=10)
    model.save("small_conv_improved.h5")


def test_model(model_file):
    images, one_hot_labels = data.load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")

    model = load_model(model_file)
    guesses = model.predict(images[:1])
    print(guesses)

if __name__ == "__main__":
    build_small_conv_model()
