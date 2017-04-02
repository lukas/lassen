import keras
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
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



if __name__ == "__main__":
    images, one_hot_labels = data.load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
    test_images, one_hot_test_labels = data.load_mnist("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte")
    #display_images(images, labels, 28, 28)
    num_classes = 10
    num_images = images.shape[0]
    num_pixels = images.shape[1]

    num_test_images = test_images.shape[0]


    images = images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    images /= 255.0
    test_images /= 255.0

    model=Sequential()
    model.add(Dense(num_classes, activation='softmax', input_shape=(num_pixels,), name="main" ))

    #weights = weights.load_weights_from_tensorflow("./tensorflow-checkpoint")
    #bias = weights.load_bias_from_tensorflow("./tensorflow-checkpoint")

    model.compile(optimizer="sgd", verbose=2, loss= 'categorical_crossentropy', metrics=['accuracy'], batch_size=100, epochs=2)

    model.fit(images, one_hot_labels, validation_data=(test_images, one_hot_test_labels))
    model.save('perceptron.h5')
    model.save_weights('perceptron-weights.h5')
