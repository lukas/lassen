
from tensorflow.python import pywrap_tensorflow
import h5py
import nets
import data

def load_weights_from_tensorflow(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    weights = reader.get_tensor('Variable')
    return weights

def load_biases_from_tensorflow(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    bias = reader.get_tensor('Variable_1')
    return bias

def load_weights_from_keras_perceptron(filename):
    f = h5py.File(filename)
    bias = f['model_weights']['dense_1']['dense_1']['bias:0'][()]
    weights = f['model_weights']['dense_1']['dense_1']['kernel:0'][()]
    return weights, bias


def load_weights_from_keras_two_layer(filename):
    f = h5py.File(filename)
    bias1 = (f['model_weights']['dense_1']['dense_1']["bias:0"][()])
    weights1 = (f['model_weights']['dense_1']['dense_1']["kernel:0"][()])
    bias0 = (f['model_weights']['main']['main']["bias:0"][()])
    weights0 = (f['model_weights']['main']['main']["kernel:0"][()])

    return weights0, bias0, weights1, bias1

def load_layer_weights_from_keras(filename, layer_name):
    f = h5py.File(filename)
    weights = (f['model_weights'][layer_name][layer_name]["kernel:0"][()])
    biases = (f['model_weights'][layer_name][layer_name]["bias:0"][()])
    return weights, biases

def set_layer_weights_keras(network, index, filename, layer_name):

    weights, biases = load_layer_weights_from_keras(filename, layer_name)

    network[index].set_layer_weights_keras(weights)
    network[index].biases = biases

    print("Set Weights", index, layer_name, weights.shape, biases.shape, network[index])

def load_three_layer_weights_keras(network, filename):
    set_layer_weights_keras(network, 0, filename, 'conv2d_1')
    set_layer_weights_keras(network, 3, filename, 'conv2d_2')
    set_layer_weights_keras(network, 6, filename, 'dense_1')
    set_layer_weights_keras(network, 8, filename, 'final')


def load_perceptron(filename):
    test_images, test_labels = data.load_test_mnist()
    network = nets.setup_perceptron(test_images, test_labels)
    weights, biases = load_weights_from_keras_perceptron(filename)
    network[0].weights = weights
    network[0].biases = biases
    return network

def load_two_layer(filename):

    test_images, test_labels = data.load_test_mnist()
    network = nets.setup_two_layer_beast()
    weights0, biases0, weights1, biases1 = load_weights_from_keras_two_layer(filename)
    print("Weight0 shape", weights0.shape)
    print("Weight1 shape", weights1.shape)
    print(weights0)
    network[0].weights = weights0
    network[0].biases = biases0
    network[2].weights = weights1
    network[2].biases = biases1
    return network

def load_small_conv(filename):
    test_images, test_labels = data.load_test_mnist()
    network = nets.setup_three_layer_mnist()

    set_layer_weights_keras(network, 0, filename, 'conv2d_1')
    set_layer_weights_keras(network, 3, filename, 'conv2d_2')
    set_layer_weights_keras(network, 6, filename, 'dense_1')
    set_layer_weights_keras(network, 8, filename, 'final')
    return network

if __name__ == "__main__":
    load_weights_from_keras("perceptron.h5")
