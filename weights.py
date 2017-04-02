from tensorflow.python import pywrap_tensorflow

def load_weights_from_tensorflow(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    weights = reader.get_tensor('Variable')
    return weights

def load_bias_from_tensorflow(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    bias = reader.get_tensor('Variable_1')
    return bias
