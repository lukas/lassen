import unittest
import mnist_lassen
import lassen
import numpy as np
import weights
import mnist_data

def manual_weight_derivative(network, images, labels, layer_index, weight_index):
    """Calculate the partial derivative of loss wrt single weight
    by changing one of the weights by epsilon and noting change in loss"""

    epsilon = 0.005
    layer = network[layer_index]
    loss, acc = lassen.gradient_batch(network, images, labels)
    layer.weights[weight_index] += epsilon
    loss2, acc = lassen.gradient_batch(network, images, labels)
    manual_gradient = (loss2 - loss) / epsilon
    # move things back
    layer.weights[weight_index] -= epsilon

    return manual_gradient

def analytic_weight_derivative(network, images, labels, layer_index, weight_index):
    """Calcuate the partial deviative of loss wrt single weight
    by backpropogation.  Should be the same as manual_derivative
    if we did our math properly"""

    loss, acc = lassen.gradient_batch(network, images, labels)
    layer = network[layer_index]
    computed_gradient = layer.weights_gradient[weight_index]

    return computed_gradient

def compare_weight_derivative(network, images, labels, layer_index, weight_index):
    m_deriv = manual_weight_derivative(network, images, labels, layer_index, weight_index)
    a_deriv = analytic_weight_derivative(network, images, labels, layer_index, weight_index)
    return abs(m_deriv - a_deriv)

def compare_all_weight_derivatives(network, images, labels):
    max_diff = -np.Inf
    for layer_index in range(len(network)):
        if network[layer_index].has_weights():
            for weight_index, weight in np.ndenumerate(network[layer_index].weights):
                diff = compare_weight_derivative(network, images, labels, layer_index, weight_index)
                max_diff = max(diff, max_diff)
                if diff > 0.001:
                    print("Discrepency in Network Layer ", layer_index, " Weight Index ", weight_index)
                    print("Manual", manual_weight_derivative(network, images, labels, layer_index, weight_index))
                    print("Analytic", analytic_weight_derivative(network, images, labels, layer_index, weight_index))
    return max_diff

def analytic_node_derivative(network, image, label, layer_index, node_index):
    activations = [image]
    for layer in network:
        activations.append(layer.forward(activations[-1]))

    gradients_rev = [label]

    for layer, activation in zip(reversed(network), reversed(activations[:-1])):
        gradients_rev.append(layer.backward(activation, gradients_rev[-1]))

    gradients=gradients_rev[::-1]
    gradient_layer = gradients[layer_index]

    return(gradients[layer_index][node_index])

def compare_node_derivative(network, image, label, layer_index, node_index):
    m_deriv = manual_node_derivative(network, image, label, layer_index, node_index)
    a_deriv = analytic_node_derivative(network, image, label, layer_index, node_index)
    return abs(m_deriv - a_deriv)

def manual_node_derivative(network, image, label, layer_index, node_index):
    """Calculate the partial derivative of loss wrt single node
    by changing one of the nodes by epsilon and noting change in loss"""

    epsilon = 0.00005

    activations = [image]
    for layer in network:
        activations.append(layer.forward(activations[-1]))

    loss = -np.dot(label, activations[-1])

    new_activations = activations[0:(layer_index+1)]

    activation_layer = activations[layer_index]
    activation_layer[node_index] += epsilon

    new_activations[layer_index] = activation_layer

    for layer in network[layer_index:]:
        new_activations.append(layer.forward(new_activations[-1]))

    loss2 = -np.dot(label, new_activations[-1])

    manual_gradient = (loss2 - loss) / epsilon

    return manual_gradient

def compare_all_node_derivatives(network, image, label):
    max_diff = -np.Inf
    for layer_index in range(len(network)):
        if network[layer_index].has_weights():
            for node_index in np.ndindex(network[layer_index].input_dim):
                diff = compare_node_derivative(network, image, label, layer_index, node_index)
                max_diff = max(diff, max_diff)
                if diff > 0.001:
                    print("Discrepency in Network Layer ", layer_index, " Weight Index ", node_index)
                    print("Manual", manual_node_derivative(network, image, label, layer_index, node_index))
                    print("Analytic", analytic_node_derivative(network, image, label, layer_index, node_index))
    return max_diff


class TestNetwork(unittest.TestCase):
    def test_dense_deriv(self):
        """Tests taking weights derivative of a dense layer + softmax layer
        (4) input = [0,1,2,3]
        (4x3) weights = [[0,1,2] ... [9,10,11]]
        (3) biases = [0,1,2]
        """

        image=np.arange(5.0 * 7.0).reshape((5, 7))
        weights = np.reshape(np.arange(5.0 * 7.0 * 3.0), (5, 7, 3))
        biases = np.arange(3)
        network=[mnist_lassen.DenseLayer((5, 7), 3), mnist_lassen.SoftmaxLayer(3)]
        network[0].set_weights(weights)
        network[0].biases=biases
        one_hot_labels = np.array([[0,1,0]])

        self.assertLess(compare_all_weight_derivatives(network, [image],
            one_hot_labels), 0.0001)
        self.assertLess(compare_all_node_derivatives(network, image,
            one_hot_labels[0]), 0.0001)


    def test_conv_deriv(self):
        """Tests the derivative of a conv layer+dense layer+softmax layer
        (4x4) input = [[0,1,2,3], ... [12,13,14,15]]
        biases = 1
        (3x3) weights = [[0,1,2], ... [6,7,8]]
        (4x4) image = [[0,1,2,3], ... [12,13,14,15]]
        label = 1
        """
        image=np.reshape(np.arange(16.),(4,4)).flatten()
        network= [lassen.ConvLayer((4,4), (3, 3), 1, 1),
                  lassen.DenseLayer(4*4,3),
                  lassen.SoftmaxLayer(3)]
        biases = np.array([1.])
        weights = np.reshape(np.arange(9.),(1,1,3,3))
        one_hot_labels = np.array([0,1,0])
        network[0].biases = biases
        network[0].set_weights(weights)
        network[1].set_weights(np.reshape(np.arange(16.*3),(16, 3)))
        one_hot_labels = np.array([[0,1,0]])


        self.assertLess(compare_all_weight_derivatives(network, [image],
             one_hot_labels), 0.0001)
        self.assertLess(compare_all_node_derivatives(network, image,
                     one_hot_labels[0]), 0.0001)

    def test_small_cnn_deriv(self):
        """Tests the derivative of a conv layer+relu+maxpool+dense layer+softmax layer
        """

        image=np.reshape(np.arange(16.),(4,4)).flatten()
        network= [lassen.ConvLayer((4,4), (3, 3), 1, 2),
                  lassen.ReluLayer(4*4*2),
                  lassen.MaxPoolLayer((4,4), (2,2), 2),
                  lassen.DenseLayer(2*2*2,3),
                  lassen.SoftmaxLayer(3)]
        biases = np.array([1.,2.])
        weights = np.reshape(np.arange(3.*3*2),(1,2,3,3))
        one_hot_labels = np.array([0,1,0])
        network[0].biases = biases
        network[0].set_weights(weights)
        network[3].set_weights(np.reshape(np.arange(4.*2*3),(4*2, 3)))
        one_hot_labels = np.array([[0,1,0]])

        self.assertLess(compare_all_weight_derivatives(network, [image],
             one_hot_labels), 0.0001)
        self.assertLess(compare_all_node_derivatives(network, image,
                     one_hot_labels[0]), 0.0001)

    def test_small_cnn(self):
        """Tests the output of a conv layer+relu+maxpool+dense layer+softmax layer
        Equivalent to in keras
           [<keras.layers.convolutional.Conv2D at 0x12cea4be0>,
            <keras.layers.pooling.MaxPooling2D at 0x12ce75630>,
            <keras.layers.core.Dense at 0x12cfbbda0>]
        """
        image=np.reshape(np.arange(16.),(4,4)).flatten()
        network= [
            lassen.ConvLayer((4,4), (3, 3), 1, 2),
            lassen.ReluLayer(4*4*2),
            lassen.MaxPoolLayer((4,4), (2,2), 2),
            lassen.DenseLayer((2,2,2),3)
        ]
        biases = np.array([1.,2.])
        weights = np.reshape(np.arange(3.*3*2),(1,2,3,3))
        one_hot_labels = np.array([0,1,0])
        network[0].biases = biases
        network[0].set_weights(weights)
        # network[3].set_weights(np.arange(4.*2*3)
        #     .reshape((2,2,2,3))
        #     .transpose((2,0,1,3)))
        #     # .reshape(2*2*2,3))
        # print("ghetto weights: ", network[3].weights)
        # print("ghetto weights sum: ", np.sum(network[3].weights))
        network[3].set_keras_weights(np.arange(4.0 * 2 * 3))

        activations = [image]
        for layer in network:
            activations.append(layer.forward(activations[-1]))

        first_layer_output= np.array([[[[   74.,   165.],
         [  122.,   285.],
         [  155.,   372.],
         [  104.,   267.]],

        [[  172.,   416.],
         [  259.,   665.],
         [  295.,   782.],
         [  187.,   539.]],

        [[  280.,   740.],
         [  403.,  1133.],
         [  439.,  1250.],
         [  271.,   839.]],

        [[  140.,   519.],
         [  188.,   783.],
         [  203.,   852.],
         [  114.,   565.]]]])

        self.assertEqual(first_layer_output.transpose(3,1,2,0).flatten().tolist(),
                            activations[1].flatten().tolist())

        second_layer_output = np.array([[[[  259.,   665.],
         [  295.,   782.]],

        [[  403.,  1133.],
         [  439.,  1250.]]]])

        self.assertEqual(second_layer_output.transpose(3,1,2,0).flatten().tolist(),
                            activations[3].flatten().tolist())

        third_layer_output = np.array([[ 66786.,  72012.,  77238.]])

        self.assertEqual(third_layer_output.flatten().tolist(),
                            activations[4].flatten().tolist())

    def test_single_conv_layer(self):
        """Tests a convolutional layer
        (4x4) input = [[0,1,2,3], ... [12,13,14,15]]
        biases = 1
        (3x3) weights = [[0,1,2], ... [6,7,8]]
        sum of output should be 3406"""

        image=np.reshape(np.arange(16),(4,4))
        network= [mnist_lassen.ConvLayer((4,4), (3, 3), 1, 1)]
        biases = np.array([1.])
        weights = np.reshape(np.arange(9),(1,1,3,3))
        network[0].biases = biases
        network[0].set_weights(weights)

        output = lassen.forward(network, image.flatten())
        true_sum = 3406.0
        my_sum = np.sum(output)

        self.assertEqual(my_sum, true_sum)

    def test_single_max_pool_layer(self):
        """Tests a max pool layer
        (4x4x2) input = [[[0,1,2,3], ... [12,13,14,15]],
                         [[16,17,18,19] ... [28, 29, 30, 31]]]
        (2x2) Max pool
        sum of output should be 164"""

        image=np.reshape(np.arange(0,32),(4,4,2))
        image=np.transpose(image, (2,0,1))
        network=[mnist_lassen.MaxPoolLayer((4,4), (2,2), 2)]
        output = lassen.forward(network, image.flatten())

        my_sum = np.sum(output)
        true_sum=164.0
        self.assertEqual(my_sum, true_sum)

    def test_single_dense_layer(self):
        """Tests a dense layer
        (4) input = [0,1,2,3]
        (4x3) weights = [[0,1,2] ... [9,10,11]]
        (3) biases = [0,1,2]
        sum of output should be 147"""

        image=np.arange(4)
        weights = np.reshape(np.arange(12),(4,3))
        biases = np.arange(3)
        network=[mnist_lassen.DenseLayer(4, 3)]
        network[0].set_weights(weights)
        network[0].biases=biases
        output = lassen.forward(network, image.flatten())
        my_sum = np.sum(output)
        true_sum=147.0
        self.assertEqual(my_sum, true_sum)


class TestKerasNetwork(unittest.TestCase):
    def test_keras_perceptron(self):
        test_images, test_labels = mnist_data.load_test_mnist()
        network = weights.load_perceptron('models/perceptron.h5')
        acc = lassen.accuracy(network, test_images, test_labels)
        self.assertTrue(np.allclose(acc, 0.88, atol = 1e-2))

    def test_keras_two_layer(self):
        test_images, test_labels = mnist_data.load_test_mnist()
        network = weights.load_two_layer('models/two_layer.h5')
        acc = lassen.accuracy(network, test_images, test_labels)
        self.assertTrue(np.allclose(acc, 0.9519, atol = 1e-2))

    def test_keras_small_conv(self):
        test_images, test_labels = mnist_data.load_test_mnist()
        network = weights.load_small_conv('models/small_conv_improved.h5')

        acc = lassen.accuracy(network, test_images[:100], test_labels[:100])
        self.assertTrue(np.allclose(acc, 0.96, atol = 1e-2))

if __name__ == '__main__':
    unittest.main()
