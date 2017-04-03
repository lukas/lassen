
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

def load_weights_from_keras(filename):
    f = h5py.File(filename)
    bias1 = (f['model_weights']['dense_1']['dense_1']["bias:0"][()])
    weights1 = (f['model_weights']['dense_1']['dense_1']["kernel:0"][()])
    bias0 = (f['model_weights']['main']['main']["bias:0"][()])
    weights0 = (f['model_weights']['main']['main']["kernel:0"][()])

    return bias0, weights0, bias1, weights1


if __name__ == "__main__":
    load_weights_from_keras("perceptron.h5")
