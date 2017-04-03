import data, weights
import numpy as np
from scipy.signal import convolve2d
import sys

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
        # print('pool_shape', self.pool_shape, type(self.pool_shape[0]))
        # print('output_shape', self.output_shape, type(self.output_shape[0]))
        # print('channels', self.channels)
        # np.set_printoptions(threshold=10000)

        # # debug - begin - old max pool
        # old_max_pool_ouput = input.reshape(
        #     self.channels,
        #     self.output_shape[0],
        #     self.pool_shape[0],
        #     self.output_shape[1],
        #     self.pool_shape[1]
        # ).max(axis=(2,4)).flatten()
        # print('old_max_pool_ouput', old_max_pool_ouput[3000:3010], old_max_pool_ouput.shape)
        # # debug - end

        # print('input', input.reshape(self.channels,)[0,12:16,12:16])
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
        assert not hasattr(self, 'max_pool_indices')
        self.max_pool_indices = input.argmax(axis=1)
        new_max_pool_output = input[
            range(self.channels * self.output_shape[0] * self.output_shape[1]),
            self.max_pool_indices
        ]

        # print('max_pool_indices', self.max_pool_indices[3000:3010])
        # print('new_max_pool_output', new_max_pool_output[3000:3010], new_max_pool_output.shape)
        #
        # print("input", input)
        # print("input.shape", input.shape)
        # print('new_max_pool_output.shape', new_max_pool_output.shape)
        return new_max_pool_output
        # y = x.reshape((2, 2, 4, 2)).transpose().reshape(8,4)
        #


    def backward(self, activations, gradient):
        # print('backward')
        # print('activations', activations.shape)
        # print('gradient', gradient.shape)
        # print('self.max_pool_indices', self.max_pool_indices.shape)
        # print("prev grad dim", (
        #     self.channels * self.output_shape[0] * self.output_shape[1],
        #     self.pool_shape[0] * self.pool_shape[1]) )

        previous_gradient = np.zeros((
            self.channels * self.output_shape[0] * self.output_shape[1],
            self.pool_shape[0] * self.pool_shape[1]) )

        previous_gradient[
            range(self.channels * self.output_shape[0] * self.output_shape[1]),
            self.max_pool_indices] = \
                gradient

        previous_gradient = previous_gradient.reshape(self.channels, self.output_shape[0],
                        self.output_shape[1], self.pool_shape[0], self.pool_shape[1])
        #print(previous_gradient.shape)
        previous_gradient = previous_gradient.transpose((0,1,3,2,4)).flatten()


        grad3d = gradient.reshape(
            self.channels,
            self.output_shape[0],
            self.output_shape[1]
        )

        activ3d = activations.reshape(
            self.channels,
            self.input_shape[0],
            self.input_shape[1]
        )

        prev3d = previous_gradient.reshape(
            self.channels,
            self.input_shape[0],
            self.input_shape[1]
        )
        #
        # print("Activation", activ3d[0:1,0:2,0:2])
        # print("Gradient Next", grad3d[0,0,0])
        # print("Gradient Prev", prev3d[0:1,0:2,0:2])

        del self.max_pool_indices

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

        # debug - begin
        print("input_dim", self.input_dim)
        print("output_dim", self.output_dim)
        print("img_shape", img_shape)
        print("kernel_shape", kernel_shape)
        print("weights shape", self.weights.shape)
        print("biases shape", self.biases.shape)
        print()
        # debug - end

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

        # print('ConvLayer - forward')
        # print('input', input)
        # print('self.input_dim', self.input_dim)
        assert input.shape == self.input_dim

        input = input.reshape((input_channels,) + self.img_shape)
        output = np.zeros((output_channels,) + self.img_shape)

        for input_index, input_channel in enumerate(input):
            for output_index, output_channel in enumerate(output):
                output_channel += \
                    convolve2d(
                        input_channel,
                        self.weights[input_index, output_index],
                        mode='same'
                    )

        output += self.biases.reshape((-1, 1, 1))

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
        activations = activations.reshape((input_channels,) + self.img_shape)
        gradient = gradient.reshape((output_channels,) + self.img_shape)

        #calculate biases one liner
        self.biases_gradient += gradient.sum(axis=(1,2))

        # weights gradient
        previous_gradient = np.zeros(activations.shape)

        for act_index, activation_channel in enumerate(activations):
            for grad_index, gradient_channel in enumerate(gradient):
                row_offset = (kernel_shape[0]//2)
                col_offset = (kernel_shape[1]//2)
                for i in range(-row_offset,row_offset+1):
                    for j in range(-col_offset,col_offset+1):
                        self.weights_gradient[act_index, grad_index, i+row_offset, j+col_offset] += np.dot(
                            activation_channel[
                                max(-i, 0):min(rows-i, rows),
                                max(-j, 0):min(cols-j, cols)].flat,
                            gradient_channel[
                                max(i, 0):min((rows + i), rows),
                                max(j, 0):min((cols + j), cols)].flat)

                previous_gradient[act_index] += convolve2d(
                    gradient_channel,
                    self.weights[act_index, grad_index, ::-1, ::-1],
                    mode='same'
                )

        return previous_gradient.reshape((-1,))


def setup_layers_perceptron(images, labels):
    layer0 = DenseLayer(images.shape[1], labels.shape[1])
    layer1 = SoftmaxLayer(labels.shape[1])
    network = [layer0, layer1]
    return network

def setup_layers_two_layer_beast(images, labels):
    intermediate_layer_size = 50
    return [
        DenseLayer(images.shape[1], intermediate_layer_size),
        ReluLayer(intermediate_layer_size),
        DenseLayer(intermediate_layer_size, labels.shape[1]),
        SoftmaxLayer(labels.shape[1]),
    ]

def setup_three_layer_with_conv():
    # channels0 = 2
    # channels1 = 2
    # neurons2 = 2
    # network = [
    #     ConvLayer((28,28), (5, 5), 1, channels0),
    #     ReluLayer(28 * 28 * channels0),
    #     MaxPoolLayer((28, 28), (14, 14), channels0),
    #
    #     # ConvLayer((14,14), (5, 5), channels0, channels1),
    #     # ReluLayer(14 * 14 * channels1),
    #     # MaxPoolLayer((14, 14), (2, 2), channels1),
    #     #
    #     # DenseLayer(7 * 7 * channels1, neurons2),
    #     # ReluLayer(neurons2),
    #     #
    #     # DenseLayer(neurons2, 10),
    #     # SoftmaxLayer(10),
    # ]
    # assert_layer_dimensions_align(network)
    # return network

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
    print_num_params(network)
    exit()
    return network

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
    return input

def gradient(network, image, label):
    activations = [image]

    for layer in network:
        activations.append(layer.forward(activations[-1]))

    loss = -np.dot(label, activations[-1])
    gradient = label

    for layer, activation in zip(reversed(network), reversed(activations[:-1])):
        gradient = layer.backward(activation, gradient)

    return loss

def gradient_batch(network, images, labels):
    for layer in network:
        layer.reset_gradient()

    loss = 0
    for image, label in zip(images, labels):
        loss += gradient(network, image, label)

    return loss

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
    num_epochs = 3
    batch_size = 100
    learn_rate = 0.001
    num_labels = labels.shape[0]

    for _ in range(num_epochs):
        rand_indices = np.random.permutation(num_labels)
        num_batches = int(num_labels/batch_size)
        #print("Num Batches", num_batches)
        for ridx in range(num_batches):
          rand_idx = rand_indices[(ridx*batch_size):(ridx*(batch_size+1))]

          batch_labels = labels[rand_idx,:]
          batch_images = images[rand_idx,:]

          l = gradient_batch(network, batch_images, batch_labels)
          for layer in network:
            layer.step(learn_rate * (1.0/batch_size))

          sys.stdout.write("Loss: %.3f  \r" % (l) )
          sys.stdout.flush()

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

def test_gradient(network, images, labels):
    epsilon = 0.0005
    layer = network[0]
    loss = gradient_batch(network, images[:1], labels[:1])
    max_gradient_index = np.unravel_index(
        np.argmax(np.abs(layer.weights_gradient)),
        layer.weights_gradient.shape
    )

    print("Max Gradient Index", max_gradient_index)
    print("New Gradient", layer.weights_gradient[max_gradient_index])

    layer.weights[max_gradient_index] += epsilon
    loss2 = gradient_batch(network, images[:1], labels[:1])
    print("Manual Gradient", (loss2 - loss) / epsilon)

    # move things back
    layer.weights[max_gradient_index] -= epsilon


def main():
    # ConvLayer((100,400), (5, 5), 32, 64)
    # return

    images, labels = data.load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
    test_images, test_labels = data.load_mnist("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte")
    images = images / 255.0
    test_images = test_images / 255.0


    # tensorflow_weights = weights.load_weights_from_tensorflow("./tensorflow-checkpoint")
    # tensorflow_biases = weights.load_biases_from_tensorflow("./tensorflow-checkpoint")

    # bias0, weights0, bias1, weights1 = weights.load_weights_from_keras('perceptron.h5')

    # network = setup_layers_perceptron(images, labels)
    # network = setup_layers_two_layer_beast(images, labels)
    network = setup_three_layer_with_conv()
    set_random_weights(network)

    # print("Loss new network", gradient_batch(network, images[:1], labels[:1]))
    # print("Loss old network", gradient_batch(old_network, images[:1], labels[:1]))
    test_gradient(network, images, labels)
    #
    # print("New Network Conv Gradient", network[0].weights_gradient)
    # print("Old Network Conv Gradient", old_network[0].weights_gradient)



    exit()
    #t1 = datetime.datetime.now()
    sgd(network, images, labels, test_images, test_labels)
    #t2 = datetime.datetime.now()

    print("Time New", t2-t1)

    #sgd(network, images[:1000], labels[:1000], test_images, test_labels)


if __name__ == "__main__":
    main()
