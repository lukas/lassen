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

        self.biases_gradient += gradient.sum(axis=(1,2))

        # weights gradient
        previous_gradient = np.zeros(activations.shape)

        for act_index, activation_channel in enumerate(activations):
            for grad_index, gradient_channel in enumerate(gradient):
                for i in range(-kernel_shape[0]//2,(kernel_shape[0]//2+1)):

                    activation_min_row = max(i, 0)
                    activation_max_row = min((rows + i), rows)
                    gradient_min_row = max(-i, 0)
                    gradient_max_row = min(rows-i, rows)

                    for j in range(-kernel_shape[1]//2,(kernel_shape[1]//2+1)):
                        activation_min_col = max(j, 0)
                        activation_max_col = min((cols + j), cols)
                        gradient_min_col = max(-j, 0)
                        gradient_max_col = min(cols-j, cols)

                        self.weights_gradient[act_index, grad_index, i, j] += np.dot(
                            activation_channel[
                                activation_min_row:activation_max_row,
                                activation_min_col:activation_max_col].flat,
                            gradient_channel[
                                gradient_min_row:gradient_max_row,
                                gradient_min_col:gradient_max_col].flat)


                for indx, kernel in enumerate(np.eye(np.prod(kernel_shape))):
                    kernel = kernel.reshape(kernel_shape)
                    kernel_index = (indx // kernel_shape[1], indx % kernel_shape[1])
                    weights_index = (act_index, grad_index) + kernel_index
                    #self.weights_gradient[weights_index] += np.dot(
                    #    gradient_channel.flat,
                    #    convolve2d(activation_channel, kernel, mode='same').flat
                    #)

                    previous_gradient[act_index] += \
                        convolve2d(
                            gradient_channel,
                            kernel * self.weights[
                                act_index,
                                grad_index,
                                kernel_shape[0] - kernel_index[0] - 1,
                                kernel_shape[1] - kernel_index[1] - 1
                            ],
                            mode='same'
                        )

                # # print('previous_gradient.shape', previous_gradient.shape)
                # # print('act_index', act_index)
                # # print('grad_index', grad_index)
                # previous_gradient[act_index,:] += \
                #     convolve2d(
                #         gradient_channel,
                #         self.weights[act_index, grad_index],
                #         mode='same'
                #     )

        # propogate the gradient backwards
        return previous_gradient.reshape((-1,))

class OldConvLayer(Layer):
    def __init__(self, img_shape, kernel_shape, input_channels, output_channels):
        """
        To convert 32 channels of size 100x400 into 64 channels with a 5x5 kernel:

        # ConvLayer((100,40), (5, 5), 32, 64)
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

        # # debug - begin
        # print("CHECKING DIMS", self)
        # print("weights.shape", self.weights.shape)
        # print("input_channels", input_channels)
        # print("output_channels", output_channels)
        # print("img_shape", self.img_shape)
        # print("input.shape", input.shape)
        # print("input_dim", self.input_dim)
        # # debug - end

        assert input.shape == self.input_dim

        input = input.reshape((input_channels,) + self.img_shape)
        output = np.zeros((output_channels,) + self.img_shape)

        # for input_index, (input_channel, weights_for_input) in enumerate(zip(input, self.weights)):
        #     for output_index, (output_channel, kernel) in enumerate(zip(output, weights_for_input)):
        #         # # debug - begin
        #         # print("computing channel %i->%i on %s->%s through %s with %s" % (
        #         #     input_index,
        #         #     output_index,
        #         #     input_channel.shape,
        #         #     output_channel.shape,
        #         #     convolve2d(input_channel, kernel, mode='same').shape,
        #         #     kernel.shape))
        #         # # debug - end
        #
        #         output_channel += convolve2d(input_channel, kernel, mode='same')

        for input_index, input_channel in enumerate(input):
            for output_index, output_channel in enumerate(output):
                output_channel += \
                    convolve2d(
                        input_channel,
                        self.weights[input_index, output_index],
                        mode='same'
                    )

        # print("output.shape", output.shape)
        # print("biases.shape", self.biases.shape)
        output += self.biases.reshape((-1, 1, 1))
        # import sys
        # sys.exit(-1)

        output = output.flatten()
        assert output.shape == self.output_dim
        return output

    def backward(self, activations, gradient):
        assert gradient.shape == self.output_dim
        assert activations.shape == self.input_dim

        # unpack the gradient and activations
        input_channels, output_channels = self.weights.shape[:2]
        kernel_shape = self.weights.shape[-2:]
        activations = activations.reshape((input_channels,) + self.img_shape)
        gradient = gradient.reshape((output_channels,) + self.img_shape)

        # # debug - begin
        # print("backward input_channels", input_channels)
        # print("backward output_channels", output_channels)
        # print("backward kernel_shape", kernel_shape)
        # # debug - end

        # biases gradient
        # print("biases_gradient.shape", self.biases_gradient.shape)
        # print("gradient.shape", gradient.shape)
        # print("gradient.sum(axis=(1,2)).shape", gradient.sum(axis=(1,2)).shape)
        # import sys
        # sys.exit(-1)
        self.biases_gradient += gradient.sum(axis=(1,2))

        # weights gradient
        previous_gradient = np.zeros(activations.shape)
        for act_index, activation_channel in enumerate(activations):
            for grad_index, gradient_channel in enumerate(gradient):
                for indx, kernel in enumerate(np.eye(np.prod(kernel_shape))):
                    kernel = kernel.reshape(kernel_shape)
                    kernel_index = (indx // kernel_shape[1], indx % kernel_shape[1])
                    weights_index = (act_index, grad_index) + kernel_index
                    self.weights_gradient[weights_index] += np.dot(
                        gradient_channel.flat,
                        convolve2d(activation_channel, kernel, mode='same').flat
                    )

                    previous_gradient[act_index] += \
                        convolve2d(
                            gradient_channel,
                            kernel * self.weights[
                                act_index,
                                grad_index,
                                kernel_shape[0] - kernel_index[0] - 1,
                                kernel_shape[1] - kernel_index[1] - 1
                            ],
                            mode='same'
                        )

                # # print('previous_gradient.shape', previous_gradient.shape)
                # # print('act_index', act_index)
                # # print('grad_index', grad_index)
                # previous_gradient[act_index,:] += \
                #     convolve2d(
                #         gradient_channel,
                #         self.weights[act_index, grad_index],
                #         mode='same'
                #     )

        # propogate the gradient backwards
        return previous_gradient.reshape((-1,))

# def log_softmax(w):
#     assert len(w.shape) == 1
#     max_weight = np.max(w, axis=0)
#     rightHandSize = np.log(np.sum(np.exp(w - max_weight), axis=0))
#     return w - (max_weight + rightHandSize)

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

def old_setup_three_layer_with_conv():
    intermediate_layer_size = 50
    intermediate_channels = 1
    network = [
        #OldConvLayer((28,28), (5, 5), 1, intermediate_channels),
        #ReluLayer(28 * 28 * intermediate_channels),
        #OldConvLayer((28,28), (5, 5), intermediate_channels, intermediate_channels),
        #ReluLayer(28 * 28 * intermediate_channels),
        OldConvLayer((28,28), (1, 1), 1, 1),
        ReluLayer(28 * 28 * intermediate_channels),

        DenseLayer(28 * 28 * intermediate_channels, intermediate_layer_size),
        ReluLayer(intermediate_layer_size),
        DenseLayer(intermediate_layer_size, 10),
        SoftmaxLayer(10),
    ]
    assert_layer_dimensions_align(network)
    return network

def setup_three_layer_with_conv():
    intermediate_layer_size = 50
    intermediate_channels = 1
    network = [
        # ConvLayer((28,28), (5, 5), 1, intermediate_channels),
        # ReluLayer(28 * 28 * intermediate_channels),
        # ConvLayer((28,28), (5, 5), intermediate_channels, intermediate_channels),
        # ReluLayer(28 * 28 * intermediate_channels),
        ConvLayer((28,28), (1, 1), 1, 1),
        ReluLayer(28 * 28 * intermediate_channels),

        DenseLayer(28 * 28 * intermediate_channels, intermediate_layer_size),
        ReluLayer(intermediate_layer_size),
        DenseLayer(intermediate_layer_size, 10),
        SoftmaxLayer(10),
    ]
    assert_layer_dimensions_align(network)
    return network

def assert_layer_dimensions_align(network):
    output_dim = network[0].output_dim
    print(network[0])
    for layer in network[1:]:
        input_dim = layer.input_dim
        print("   %s == %s" % (input_dim, output_dim))
        print(layer)
        assert input_dim == output_dim, "%s != %s" % (input_dim, output_dim)
        output_dim = layer.output_dim

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
    num_epochs = 100
    batch_size = 1000
    learn_rate = 0.01
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
        test_gradient(network, images, labels)

def set_random_weights(network):
    for layer in network:
        if hasattr(layer, 'weights'):
            layer.weights = np.random.normal(size=layer.weights.shape)
            layer.biases = np.random.normal(size=layer.biases.shape)

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
    loss = gradient_batch(network, images[1:5], labels[1:5])
    max_gradient_index = np.unravel_index(
        np.argmax(layer.weights_gradient),
        layer.weights_gradient.shape
    )

    print("Max Gradient Index", max_gradient_index)
    print("New Gradient", layer.weights_gradient[max_gradient_index])

    layer.weights[max_gradient_index] += epsilon
    loss2 = gradient_batch(network, images[1:5], labels[1:5])
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
    old_network = old_setup_three_layer_with_conv()

    set_random_weights(old_network)
    copy_weights(old_network, network)

    print(gradient_batch(network, images[:1], labels[:1]))
    print(gradient_batch(old_network, images[:1], labels[:1]))
    test_gradient(network, images, labels)
    test_gradient(old_network, images, labels)

    assert np.array_equal(network[0].weights, old_network[0].weights)
    assert np.array_equal(network[0].biases, old_network[0].biases)
    exit()
    # network[0].biases = bias0
    # network[0].weights = weights0
    # network[2].biases = bias1
    # network[2].weights = weights1


    #print(accuracy(network,images,labels))
    #print(forward(network,images[0]))

    #set_random_weights(network)
    #sgd(network, images, labels, test_images, test_labels)


if __name__ == "__main__":
    main()
