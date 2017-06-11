import mnist_data, weights, mnist_lassen
import numpy as np

import ast
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

    def set_weights(self, weights):
        assert False, "%s doesn't take weights." % type(self).__name__

    def set_keras_weights(self, keras_weights):
        assert False, "%s doesn't take weights." % type(self).__name__

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
        self.weights = np.zeros(self.input_dim + self.output_dim)
        self.biases = np.zeros(self.output_dim)
        self.input_elts = np.product(self.input_dim)

    def __str__(self):
        return "DenseLayer [%s -> %s]" % (
            self.input_dim,
            self.output_dim
        )

    def forward(self, input):
        return np.dot(self.weights
            .reshape((self.input_elts, -1))
            .T, input.flat) + self.biases

    def backward(self, activations, gradient):
        assert gradient.shape == self.output_dim
        assert activations.shape == self.input_dim

        self.biases_gradient += gradient
        self.weights_gradient += \
            np.outer(activations.flat, gradient).reshape(
                self.input_dim + self.output_dim)

        return np.dot(self.weights, gradient)

    def reset_gradient(self):
        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

    def set_weights(self, weights):
        assert weights.shape == self.input_dim + self.output_dim, \
            "Setting weights of dim %s which should be %s." % \
            (str(weights.shape), str(self.input_dim + self.output_dim))
        self.weights = weights

    def set_keras_weights(self, keras_weights):
        keras_shape = \
            self.input_dim[1:] + (self.input_dim[0],) + self.output_dim
        input_axes = len(self.input_dim)
        axis_order = \
            (input_axes - 1,) + tuple(range(input_axes - 1)) + (input_axes,)
        weights = keras_weights.reshape(keras_shape).transpose(axis_order)
        self.set_weights(weights)

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
        self.output_shape = tuple(np.divide(input_shape, pool_shape)
            .astype(np.int32))
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
        mpi = input.reshape(
            self.channels,
            self.input_shape[0],
            self.input_shape[1])

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
        assert kernel_shape[0] % 2 == 1
        assert kernel_shape[1] % 2 == 1
        super().__init__(
            np.prod(img_shape) * input_channels,
            np.prod(img_shape) * output_channels
        )
        self.img_shape = img_shape
        self.input_channels = input_channels
        self.kernel_shape = kernel_shape
        self.output_channels = output_channels

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
        #input_channels, output_channels = self.weights.shape[:2]
        assert input.shape == self.input_dim, \
            "Input shape %s different than expected dimension %s" % \
                (input.shape, self.input_dim)

        input = input.reshape((self.input_channels,) + self.img_shape)
        output = np.zeros((self.output_channels,) + self.img_shape)

        for input_index, input_channel in enumerate(input):
            for output_index, output_channel in enumerate(output):
                output_channel += \
                    convolve(
                        input_channel,
                        self.weights[input_index, output_index],
                        mode='same'
                    )

        output += self.biases.reshape((-1, 1, 1))

        #output = output.transpose(1,2,0)

        output = output.flatten()

        assert output.shape == self.output_dim
        return output

    def backward(self, activations, gradient):
        assert gradient.shape == self.output_dim
        assert activations.shape == self.input_dim

        # unpack the gradient and activations
        rows, cols = self.img_shape
        #input_channels, output_channels = self.weights.shape[:2]
        #kernel_shape = self.weights.shape[-2:]
        row_offset = self.kernel_shape[0] // 2
        col_offset = self.kernel_shape[1] // 2
        activations = activations.reshape((self.input_channels,) + self.img_shape)
        gradient = gradient.reshape((self.output_channels,) + self.img_shape)

        # calculate biases
        self.biases_gradient += gradient.sum(axis=(1,2))

        # weights gradient
        for i in range(-row_offset,row_offset+1):
            for j in range(-col_offset,col_offset+1):
                activations_subarray = activations[:,
                    max(-i, 0):min(rows-i, rows),
                    max(-j, 0):min(cols-j, cols)
                ].reshape((self.input_channels, -1))
                gradient_subarray = gradient[:,
                    max(i, 0):min((rows + i), rows),
                    max(j, 0):min((cols + j), cols)
                ].reshape((self.output_channels, -1)).T
                self.weights_gradient[:,:,i+row_offset,j+row_offset] += \
                    np.dot(activations_subarray, gradient_subarray)

        self.weights_gradient = self.weights_gradient[:,:,::-1,::-1]

        # previous gradient
        previous_gradient = np.zeros(activations.shape)
        for act_index, activation_channel in enumerate(activations):
            for grad_index, gradient_channel in enumerate(gradient):
                previous_gradient[act_index] += convolve(
                    gradient_channel,
                    self.weights[act_index, grad_index, ::-1, ::-1],
                    mode='same'
                )


        #previous_gradient.transpose(1,2,0)
        previous_gradient = previous_gradient.reshape((-1,))


        return previous_gradient

    def set_weights(self, weights):

        assert(weights.shape ==  (self.input_channels, self.output_channels) \
                                  + self.kernel_shape)
        self.weights = weights

    def set_keras_weights(self, keras_weights):
        self.set_weights(keras_weights.transpose(2,3,0,1))

class LSTMLayer(Layer):
    def __init__(self, input_dim, hidden_dim, timesteps):
        super().__init__(input_dim, hidden_dim)

        # additional dimensi
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps

        # weights
        self.w_i = np.zeros(input_dim, hidden_states)
        self.w_f = np.zeros(input_dim, hidden_states)
        self.w_c = np.zeros(input_dim, hidden_states)
        self.w_o = np.zeros(input_dim, hidden_states)

        # recurrent weights
        self.u_i = np.zeros(hidden_states, hidden_states)
        self.u_f = np.zeros(hidden_states, hidden_states)
        self.u_c = np.zeros(hidden_states, hidden_states)
        self.u_o = np.zeros(hidden_states, hidden_states)

        # biases
        self.b_i = np.zeros(hidden_states)
        self.b_f = np.zeros(hidden_states)
        self.b_c = np.zeros(hidden_states)
        self.b_o = np.zeros(hidden_states)

        # reset the state
        self.reset_state()

    def __str__(self):
        return "LSTMLayer (%s %s %s)[%s -> %s]" % (
            self.input_dim,
            self.hidden_dim,
            self.timesteps,
            self.input_dim,
            self.output_dim
        )

    def reset_state(self):
        """Reset the state of h and c."""
        self.h_old = np.zeros(hidden_states)
        self.c_old = np.array(h_old)

    def forward(self, input):
        i = sigmoid(np.dot(w_i.T, x) + np.dot(u_i, h_old)+b_i)
        c_tilde = np.tanh(np.dot(w_c.T, x)+np.dot(u_c, h_old)+b_c)
        f = sigmoid(np.dot(w_f.T, x) + np.dot(u_f, h_old)+b_f)
        c_new = i * c_tilde + f * c_old
        o = sigmoid(np.dot(w_o.T, x) + np.dot(u_o, h_old)+b_o)
        h_new = o * np.tanh(c_new)

    def forward_onestep(self, input):
        """Advances the lstm forward one step, statefully."""


    def backward(self, activations, gradient):
        assert gradient.shape == self.output_dim
        assert activations.shape == self.input_dim

        self.biases_gradient += gradient
        self.weights_gradient += \
            np.outer(activations.flat, gradient).reshape(
                self.input_dim + self.output_dim)

        return np.dot(self.weights, gradient)

    def reset_gradient(self):
        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

    def set_weights(self, weights):
        assert weights.shape == self.input_dim + self.output_dim, \
            "Setting weights of dim %s which should be %s." % \
            (str(weights.shape), str(self.input_dim + self.output_dim))
        self.weights = weights

    def set_keras_weights(self, keras_weights):
        keras_shape = \
            self.input_dim[1:] + (self.input_dim[0],) + self.output_dim
        input_axes = len(self.input_dim)
        axis_order = \
            (input_axes - 1,) + tuple(range(input_axes - 1)) + (input_axes,)
        weights = keras_weights.reshape(keras_shape).transpose(axis_order)
        self.set_weights(weights)


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
@click.option('--kernel_size', default='1')
@click.option('--translate', default='(0,0)')
def test_identity(kernel_size, translate):
    # parse the arguments
    kernel_size = int(kernel_size)
    translate = ast.literal_eval(translate)
    assert -(kernel_size // 2) <= translate[0] <= kernel_size // 2
    assert -(kernel_size // 2) <= translate[1] <= kernel_size // 2


    # create the network
    conv_layer = ConvLayer((28,28), (kernel_size, kernel_size), 1, 1)
    conv_layer.weights[
        0, 0,
        kernel_size // 2 - translate[0],
        kernel_size // 2 - translate[1]
    ] = 1.0

    # load the images and run the newtwork
    test_image = mnist_data.load_train_mnist()[0][0].astype(np.float64)
    out_image = conv_layer.forward(test_image)

    # compute the right answer (analytically)
    right_answer = test_image.copy()
    print("right answer", right_answer.shape, right_answer.sum())

    # assert that it works properly
    print("Testing %s -> %s..." % (test_image.shape, out_image.shape))
    print("Kernel Size:       %s" % kernel_size)
    print("Translate:         %s" % (translate,))
    print("Sum Input:         %s" % np.sum(test_image))
    print("Sum Right Answer:  %s" % np.sum(right_answer))
    print("Sum Output:        %s" % np.sum(out_image))
    print("All close:         %s" % np.allclose(test_image, out_image))
    print("Array Equal:       %s" % np.array_equal(test_image, out_image))
    print("Max Difference:    %s" % np.max(np.abs(test_image - out_image)))
    print("Shape Input:       %s" % test_image.shape)
    print("Shape Output:      %s" % out_image.shape)



@cli.command()
def network_output():
    pass

@cli.command()
def three_layer_accuracy():
    images, labels = mnist_data.load_train_mnist()
    test_images, test_labels = mnist_data.load_test_mnist()
    network = mnist_lassen.load_network('three_layer_mnist', images, labels)
    weights.load_three_layer_weights_keras(network, 'models/small_conv_improved.h5')
    print("Second convnet weights", network[3].weights.shape)
    print("Second convnet weight", network[3].weights[0,0,:,:])


    print(forward(network, images[0]))
    #print(accuracy(network, images[:20], labels[:20]))

@cli.command()
def run():
    images, labels = mnist_data.load_train_mnist()
    test_images, test_labels = mnist_data.load_test_mnist()

    network = setup_three_layer_with_conv()
    set_random_weights(network)

    weights.load_three_layer_weights_keras(network, 'models/small_conv.h5')

    sgd(network, images, labels, test_images[:100], test_labels[:100])



if __name__ == "__main__":
    cli()
