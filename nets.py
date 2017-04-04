
from dnn_homebrew import DenseLayer, SoftmaxLayer, \
    ReluLayer, MaxPoolLayer, ConvLayer, assert_layer_dimensions_align

def setup_perceptron(images, labels):
    layer0 = DenseLayer(images.shape[1], labels.shape[1])
    layer1 = SoftmaxLayer(labels.shape[1])
    network = [layer0, layer1]
    return network


def setup_two_layer_beast(images, labels):
    intermediate_layer_size = 50
    return [
        DenseLayer(images.shape[1], intermediate_layer_size),
        ReluLayer(intermediate_layer_size),
        DenseLayer(intermediate_layer_size, labels.shape[1]),
        SoftmaxLayer(labels.shape[1]),
    ]

def setup_three_layer_mnist():
    channels0 = 32
    channels1 = 64
    neurons2 = 1024
    network = [
        ConvLayer((28,28), (5, 5), 1, channels0),
        ReluLayer(28 * 28 * channels0),
        MaxPoolLayer((28, 28), (2, 2), channels0),

        ConvLayer((14,14), (5, 5), channels0, channels1),
        ReluLayer(14 * 14 * channels1),
        MaxPoolLayer((14, 14), (2, 2), channels1),

        DenseLayer(7 * 7 * channels1, neurons2),
        ReluLayer(neurons2),

        DenseLayer(neurons2, 10),
        SoftmaxLayer(10),
    ]

    assert_layer_dimensions_align(network)

    return network

def load_network(name, images, labels):
    if name == "perceptron":
        return setup_perceptron(images, labels)
    elif name == "two_layer_beast":
        return setup_two_layer_beast(images, labels)
    elif name == "three_layer_mnist":
        return setup_three_layer_mnist()
    else:
        raise ArgumentError("Unknown Network Name")
