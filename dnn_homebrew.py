import data, weights, nets
import numpy as np

import scipy.ndimage
import sys
import cProfile
import re
import click

class Layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if type(self.input_dim) != tuple:
            self.input_dim = (int(input_dim),)
        if type(self.output_dim) != tuple:
            self.output_dim = (int(output_dim),)

    def has_weights(self):
        return hasattr(self, 'weights') and hasattr(self, 'biases')

    def reset_gradient(self):
        if self.has_weights():
            self.weights_gradient = np.zeros(self.weights.shape)
            self.biases_gradient = np.zeros(self.biases.shape)

    def step(self, step_size):
        if self.has_weights():
            self.weights -= self.weights_gradient * step_size
            self.biases -= self.biases_gradient * step_size

class SoftmaxLayer(Layer):
    def __init__(self, input_dim):
        super().__init__(input_dim, input_dim)

    def __str__(self):
        return "SoftmaxLayer(%s) [%s -> %s]" % (
            self.input_dim[0],
            self.input_dim,
            self.output_dim
        )

    def log_softmax(self, w):
        assert len(w.shape) == 1
        max_weight = np.max(w, axis=0)
        rightHandSize = np.log(np.sum(np.exp(w - max_weight), axis=0))

        return w - (max_weight + rightHandSize)

    def forward(self, input):
        return self.log_softmax(input)

    def backward(self, input, output):
        input = self.log_softmax(input)
        return np.exp(input) - output

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.weights = np.zeros((input_dim, output_dim))
        self.biases = np.zeros(output_dim)

    def __str__(self):
        return "DenseLayer(%s, %s) [%s -> %s]" % (
            self.input_dim[0],
            self.output_dim[0],
            self.input_dim,
            self.output_dim
        )

    def forward(self, input):
        return np.dot(self.weights.T, input) + self.biases

    def backward(self, activations, gradient):
        assert gradient.shape == self.output_dim
        assert activations.shape == self.input_dim
        self.biases_gradient += gradient
        self.weights_gradient += np.outer(activations, gradient)
        return np.dot(self.weights, gradient)

    def reset_gradient(self):
        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

class ReluLayer(Layer):
    def __init__(self, input_dim):
        super().__init__(input_dim, input_dim)

    def __str__(self):
        return "ReluLayer(%s) [%s -> %s]" % (
            self.input_dim,
            self.input_dim,
            self.output_dim
        )

    def forward(self, input):
        # print("input", input)
        # print("input shape", input.shape)
        # print("with max", np.maximum(input, 0.0))
        return np.maximum(input, 0.0)

    def backward(self, activations, gradient):
        return (activations > 0.0) * gradient

class MaxPoolLayer(Layer):
    def __init__(self, input_shape, pool_shape, channels):
        # superclass constructor
        assert len(input_shape) == 2
        assert len(pool_shape) == 2
        assert input_shape[0] % pool_shape[0] == 0
        assert input_shape[1] % pool_shape[1] == 0
        self.input_shape = input_shape
        self.output_shape = tuple(np.divide(input_shape, pool_shape).astype(np.int32))
        self.pool_shape = pool_shape
        self.channels = channels
        input_dim = np.prod(self.input_shape) * channels
        output_dim = np.prod(self.output_shape) * channels
        super().__init__(input_dim, output_dim)

    def __str__(self):
        return "MaxPoolLayer(%s, %s, %i) [%s -> %s]" % (
            self.input_shape,
            self.pool_shape,
            self.channels,
            self.input_dim,
            self.output_dim
        )

    def forward(self, input):
        mpi = input.reshape(self.channels, self.input_shape[0], self.input_shape[1])

        input = input \
            .reshape((
                self.channels,
                self.output_shape[0],
                self.pool_shape[0],
                self.output_shape[1],
                self.pool_shape[1])) \
            .transpose((0,1,3,2,4)) \
            .reshape((
                self.channels * self.output_shape[0] * self.output_shape[1],
                self.pool_shape[0] * self.pool_shape[1]))
        #assert not hasattr(self, 'max_pool_indices')
        self.max_pool_indices = input.argmax(axis=1)
        new_max_pool_output = input[
            range(self.channels * self.output_shape[0] * self.output_shape[1]),
            self.max_pool_indices
        ]
        mpo = new_max_pool_output.reshape(self.channels, self.output_shape[0], self.output_shape[1])
        return new_max_pool_output

    def backward(self, activations, gradient):
        previous_gradient = np.zeros((
            self.channels * self.output_shape[0] * self.output_shape[1],
            self.pool_shape[0] * self.pool_shape[1]))

        previous_gradient[
            range(self.channels * self.output_shape[0] * self.output_shape[1]),
            self.max_pool_indices
        ] = gradient
        del self.max_pool_indices

        previous_gradient = previous_gradient \
            .reshape(
                self.channels,
                self.output_shape[0],
                self.output_shape[1],
                self.pool_shape[0],
                self.pool_shape[1]) \
            .transpose((0,1,3,2,4)) \
            .flatten()
        return previous_gradient

    def reset_gradient(self):
        super().reset_gradient()



class ConvLayer(Layer):
    def __init__(self, img_shape, kernel_shape, input_channels, output_channels):
        """
        To convert 32 channels of size 100x400 into 64 channels with a 5x5 kernel:

        ConvLayer((100,400), (5, 5), 32, 64)
        """
        # superclass constructor
        assert len(img_shape) == 2
        assert len(kernel_shape) == 2
        super().__init__(
            np.prod(img_shape) * input_channels,
            np.prod(img_shape) * output_channels
        )
        self.img_shape = img_shape

        # find some weights
        weights_shape = (input_channels, output_channels) + kernel_shape
        self.weights = np.zeros(weights_shape)
        self.biases = np.zeros(output_channels)

    def __str__(self):
        return "ConvLayer(%s, %s, %s, %s) [%s -> %s]" % (
            self.img_shape,
            self.weights.shape[-2:], # kernel_shape,
            self.weights.shape[0],   # input_channels,
            self.weights.shape[1],   # output_channels
            self.input_dim,
            self.output_dim
        )

    def forward(self, input):
        input_channels, output_channels = self.weights.shape[:2]
        assert input.shape == self.input_dim

        input = input.reshape((input_channels,) + self.img_shape)
        output = np.zeros((output_channels,) + self.img_shape)

        for input_index, input_channel in enumerate(input):
            for output_index, output_channel in enumerate(output):
                output_channel += \
                    convolve(
                        input_channel,
                        self.weights[input_index, output_index],
                        mode='same'
                    )

        output += self.biases.reshape((-1, 1, 1))

        # print("Biases", self.biases.shape)
        # print("Biases", self.biases)
        # print("Input shape",input.shape)
        # print("Input", input[0,:,:])
        # print("Output shape", output.shape)
        # print("Output", output[0, :, :])
        # print("Weights Shape", self.weights.shape)
        # print("Weights", self.weights[0,0,:,:])
        output = output.flatten()
        assert output.shape == self.output_dim
        return output

    def backward(self, activations, gradient):
        assert gradient.shape == self.output_dim
        assert activations.shape == self.input_dim

        # unpack the gradient and activations
        rows, cols = self.img_shape
        input_channels, output_channels = self.weights.shape[:2]
        kernel_shape = self.weights.shape[-2:]
        row_offset = kernel_shape[0] // 2
        col_offset = kernel_shape[1] // 2
        activations = activations.reshape((input_channels,) + self.img_shape)
        gradient = gradient.reshape((output_channels,) + self.img_shape)

        # calculate biases
        self.biases_gradient += gradient.sum(axis=(1,2))

        # weights gradient
        for i in range(-row_offset,row_offset+1):
            for j in range(-col_offset,col_offset+1):
                activations_subarray = activations[:,
                    max(-i, 0):min(rows-i, rows),
                    max(-j, 0):min(cols-j, cols)
                ].reshape((input_channels, -1))
                gradient_subarray = gradient[:,
                    max(i, 0):min((rows + i), rows),
                    max(j, 0):min((cols + j), cols)
                ].reshape((output_channels, -1)).T
                self.weights_gradient[:,:,i+row_offset,j+row_offset] += \
                    np.dot(activations_subarray, gradient_subarray)

        # previous gradient
        previous_gradient = np.zeros(activations.shape)
        for act_index, activation_channel in enumerate(activations):
            for grad_index, gradient_channel in enumerate(gradient):
                previous_gradient[act_index] += convolve(
                    gradient_channel,
                    self.weights[act_index, grad_index, ::-1, ::-1],
                    mode='same'
                )
        return previous_gradient.reshape((-1,))

def convolve(matrix, kernel, mode):
    # For some crazy reason, have to invert the kernel array
    return scipy.ndimage.convolve(matrix, kernel[::-1, ::-1], mode='constant' )


def assert_layer_dimensions_align(network):
    output_dim = network[0].output_dim
    print(network[0])
    for layer in network[1:]:
        input_dim = layer.input_dim
        print(layer)
        assert input_dim == output_dim, "%s != %s" % (input_dim, output_dim)
        output_dim = layer.output_dim


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 24, 24, 32)        832
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 8, 8, 64)          51264
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 4096)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1024)              4195328
# _________________________________________________________________
# final (Dense)                (None, 10)                10250
# =================================================================

def print_num_params(network):
    def print_params(*params):
        params = list(params)
        for index, param in enumerate(params):
            if type(param) == int:
                params[index] = format(param, ',d')
        print("%-15s %11s %11s %11s" % tuple(params))

    print_params("Layer", "Weights", "Biases", "Total")
    print("-" * 51)
    total_weights, total_biases = 0, 0
    for layer in network:
        if layer.has_weights():
            num_weights = int(np.prod(layer.weights.shape))
            num_biases = int(np.prod(layer.biases.shape))
            num_both = num_weights + num_biases
            total_weights += num_weights
            total_biases += num_biases
        else:
            num_weights, num_biases, num_both = '--', '--', '--'
        print_params(type(layer).__name__, num_weights, num_biases, num_both)
    print("-" * 51)
    print_params("TOTAL", total_weights, total_biases,
        total_weights + total_biases)

def forward(network, image):
    input = image

    for layer in network:
        input = layer.forward(input)
        print(layer)
        print("Sum", sum(input.flatten()))

    return input

def gradient(network, image, label):
    activations = [image]

    for layer in network:
        activations.append(layer.forward(activations[-1]))

    loss = -np.dot(label, activations[-1])
    acc = np.argmax(activations[-1]) == np.argmax(label)
    gradient = label

    for layer, activation in zip(reversed(network), reversed(activations[:-1])):
        gradient = layer.backward(activation, gradient)

    return loss, acc

def gradient_batch(network, images, labels):
    for layer in network:
        layer.reset_gradient()

    loss = 0
    acc = 0
    for image, label in zip(images, labels):
        im_loss, im_acc = gradient(network, image, label)
        loss+=im_loss
        acc+=im_acc

    return (loss/labels.shape[0]), (acc/labels.shape[0])

def classify(network, image):
    output = forward(network, image)
    cls = np.argmax(output)
    return cls

def accuracy(network, images, labels):
    guess = np.zeros(images.shape[0])
    for idx, image in enumerate(images):
        guess[idx] = classify(network, image)

    answer = np.argmax(labels, axis=1)
    return np.sum(np.equal(guess, answer))/len(guess)

def sgd(network, images, labels, test_images, test_labels):
    num_epochs = 10
    batch_size = 10
    learn_rate = 0.001
    num_labels = labels.shape[0]

    for _ in range(num_epochs):
        rand_indices = np.random.permutation(num_labels)
        num_batches = int(num_labels/batch_size)
        #print("Num Batches", num_batches)
        for ridx in range(num_batches):
          rand_idx = rand_indices[(ridx*batch_size):((ridx+1)*batch_size)]

          batch_labels = labels[rand_idx,:]
          batch_images = images[rand_idx,:]

          l, acc = gradient_batch(network, batch_images, batch_labels)

          for layer in network:
            layer.step(learn_rate * (1.0/batch_size))

          sys.stdout.write("Loss: %.3f Acc: %.3f \r" % (l, acc) )
          sys.stdout.flush()

        #print("Complete")

        print("Train Accuracy: %5.2f%%  -  Test Accuracy: %5.2f%%" %
            ( 100 * accuracy(network, images, labels),
              100 * accuracy(network, test_images, test_labels) ))
        #test_gradient(network, images, labels)

def set_random_weights(network):
    for layer in network:
        if hasattr(layer, 'weights'):
            layer.weights = np.abs(np.random.normal(
                scale=0.1,
                size=layer.weights.shape))
            layer.biases = np.abs(np.random.normal(
                scale=0.1,
                size=layer.biases.shape))

def set_unit_weights(network):
    for layer in network:
        if hasattr(layer, 'weights'):
            layer.weights = np.ones(layer.weights.shape)
            layer.biases = np.ones(layer.biases.shape)

def copy_weights(network1, network2 ):
    for layer1, layer2 in zip(network1, network2):
        if layer1.has_weights():
            assert(layer2.has_weights())
            layer2.weights[:] = layer1.weights
            layer2.biases[:] = layer1.biases


@click.group()
def cli():
    pass

@cli.command()
@click.option('--network_name', default='perceptron')
def test_gradient(network_name):
    images, labels = data.load_test_mnist()
    network=nets.load_network(network_name, images, labels)

    epsilon = 0.0005
    layer = network[0]
    loss, acc = gradient_batch(network, images[:1], labels[:1])
    max_gradient_index = np.unravel_index(
        np.argmax(np.abs(layer.weights_gradient)),
        layer.weights_gradient.shape
    )

    print("Computing gradient along max gradient index", max_gradient_index)

    computed_gradient = layer.weights_gradient[max_gradient_index]
    print("Computed Gradient", computed_gradient)

    layer.weights[max_gradient_index] += epsilon
    loss2, acc = gradient_batch(network, images[:1], labels[:1])
    manual_gradient = (loss2 - loss) / epsilon
    print("Manual Gradient", manual_gradient)

    # move things back
    layer.weights[max_gradient_index] -= epsilon

    assert (np.abs(manual_gradient - computed_gradient) < epsilon)

@cli.command()
def network_output():
    pass

@cli.command()
def three_layer_accuracy():
    images, labels = data.load_train_mnist()
    test_images, test_labels = data.load_test_mnist()
    network = nets.load_network('three_layer_mnist', images, labels)
    weights.load_three_layer_weights_keras(network, 'small_conv_improved.h5')
    print("Second convnet weights", network[3].weights.shape)
    print("Second convnet weight", network[3].weights[0,0,:,:])


    print(forward(network, images[0]))
    #print(accuracy(network, images[:20], labels[:20]))

@cli.command()
def run():
    images, labels = data.load_train_mnist()
    test_images, test_labels = data.load_test_mnist()

    network = setup_three_layer_with_conv()
    set_random_weights(network)

    weights.load_three_layer_weights_keras(network, 'small_conv.h5')

    sgd(network, images, labels, test_images[:100], test_labels[:100])



if __name__ == "__main__":
    cli()
