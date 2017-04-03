import data, weights
import numpy as np
import sys

class Layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if type(self.input_dim) == int:
            self.input_dim = (input_dim,)
        if type(self.output_dim) == int:
            self.output_dim = (output_dim,)

    def reset_gradient(self):
        pass

    def step(self, step_size):
        pass

class SoftmaxLayer(Layer):
    def __init__(self, input_dim):
        super().__init__(input_dim, input_dim)

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

    def forward(self, input):
        return np.dot(self.weights.T, input) + self.biases

    def backward(self, activations, gradient):
        assert gradient.shape == self.output_dim
        assert activations.shape == self.input_dim
        self.biases_gradient += gradient
        self.weight_gradient += np.outer(activations, gradient)
        return np.dot(self.weights, gradient)

    def reset_gradient(self):
        self.weight_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

    def step(self, step_size):
        self.weights -= self.weight_gradient * step_size
        self.biases -= self.biases_gradient * step_size

class ReluLayer(Layer):
    def __init__(self, input_dim):
        super().__init__(input_dim, input_dim)

    def forward(self, input):
        # print("input", input)
        # print("input shape", input.shape)
        # print("with max", np.maximum(input, 0.0))
        return np.maximum(input, 0.0)

    def backward(self, activations, gradient):
        return (activations > 0.0) * gradient

# class ConvLayer(Layer):
#     def __init__(self, input_dim, ):
#         super().__init__(input_dim, output_dim)
#         self.weights = np.zeros((input_dim, output_dim))
#         self.biases = np.zeros(output_dim)
#
#     def forward(self, input):
#         return np.dot(self.weights.T, input) + self.biases
#
#     def backward(self, activations, gradient):
#         assert gradient.shape == self.output_dim
#         assert activations.shape == self.input_dim
#         self.biases_gradient += gradient
#         self.weight_gradient += np.outer(activations, gradient)
#         return np.dot(self.weights, gradient)
#
#     def reset_gradient(self):
#         self.weight_gradient = np.zeros(self.weights.shape)
#         self.biases_gradient = np.zeros(self.biases.shape)
#
#     def step(self, step_size):
#         self.weights -= self.weight_gradient * step_size
#         self.biases -= self.biases_gradient * step_size

def log_softmax(w):
    assert len(w.shape) == 1
    max_weight = np.max(w, axis=0)
    rightHandSize = np.log(np.sum(np.exp(w - max_weight), axis=0))

    return w - (max_weight + rightHandSize)


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
    num_batches = 100
    learn_rate = 0.001
    batch_size = 100
    for _ in range(num_epochs):
        for _ in range(num_batches):
          rand_idx = np.floor(np.multiply(np.random.rand(batch_size), len(images))).astype(int)
          batch_labels = labels[rand_idx,:]
          batch_images = images[rand_idx,:]

          l = gradient_batch(network, batch_images, batch_labels)
          for layer in network:
            layer.step(learn_rate * (1.0/batch_size))

          sys.stdout.write("Loss: %.3f  \r" % (l) )
          sys.stdout.flush()

        print("Train Accuracy %.2f%% " % (100*accuracy(network, images, labels)), end="")
        print("Test Accuracy  %.2f%% " % (100*accuracy(network, test_images, test_labels)))
        # test_gradient(network, images, labels)

def set_random_weights(network):
    for layer in network:
        if hasattr(layer, 'weights'):
            layer.weights = np.random.normal(size=layer.weights.shape)

def test_gradient(network, images, labels):
    epsilon = 0.005
    loss = gradient_batch(network, images[1:5], labels[1:5])
    dense_layer = network[0]
    print("New Gradient", dense_layer.weight_gradient[200,3])

    dense_layer.weights[200,3] += epsilon
    loss2 = gradient_batch(network, images[1:5], labels[1:5])
    print("Manual Gradient", (loss2 - loss) / epsilon)

    # move things back
    dense_layer.weights[200,3] -= epsilon


def main():
    images, labels = data.load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
    test_images, test_labels = data.load_mnist("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte")
    images = images / 255.0
    test_images = test_images / 255.0

    # tensorflow_weights = weights.load_weights_from_tensorflow("./tensorflow-checkpoint")
    # tensorflow_biases = weights.load_biases_from_tensorflow("./tensorflow-checkpoint")

    # network = setup_layers_perceptron(images, labels)
    network = setup_layers_two_layer_beast(images, labels)
    set_random_weights(network)
    sgd(network, images, labels, test_images, test_labels)


if __name__ == "__main__":
    main()
