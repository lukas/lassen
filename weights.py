
from tensorflow.python import pywrap_tensorflow
import h5py

def load_weights_from_tensorflow(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    weights = reader.get_tensor('Variable')
    return weights

def load_biases_from_tensorflow(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    bias = reader.get_tensor('Variable_1')
    return bias

def load_weights_from_small_keras(filename):
    f = h5py.File(filename)
    bias1 = (f['model_weights']['dense_1']['dense_1']["bias:0"][()])
    weights1 = (f['model_weights']['dense_1']['dense_1']["kernel:0"][()])
    bias0 = (f['model_weights']['main']['main']["bias:0"][()])
    weights0 = (f['model_weights']['main']['main']["kernel:0"][()])

    return bias0, weights0, bias1, weights1

def load_layer_weights_from_keras(filename, layer_name):
    f = h5py.File(filename)
    weights = (f['model_weights'][layer_name][layer_name]["kernel:0"][()])
    biases = (f['model_weights'][layer_name][layer_name]["bias:0"][()])
    return weights, biases

def set_layer_weights_keras(network, index, filename, layer_name):

    weights, biases = load_layer_weights_from_keras(filename, layer_name)
    if (layer_name.startswith('conv')):
        weights = weights.transpose(2,3,0,1)
    network[index].weights = weights
    network[index].bias = biases

    print("Set Weights", index, layer_name, weights.shape, biases.shape, network[index])

def load_three_layer_weights_keras(network, filename):
    set_layer_weights_keras(network, 0, filename, 'conv2d_1')
    set_layer_weights_keras(network, 3, filename, 'conv2d_2')
    set_layer_weights_keras(network, 6, filename, 'dense_1')
    set_layer_weights_keras(network, 8, filename, 'final')





if __name__ == "__main__":
    load_weights_from_keras("perceptron.h5")
