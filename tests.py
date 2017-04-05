import unittest
import nets
import dnn_homebrew
import numpy as np

class TestNetwork(unittest.TestCase):

    def test_single_conv_layer(self):
        img=np.reshape(np.arange(0,16),(4,4))
        network= [nets.ConvLayer((4,4), (3, 3), 1, 1)]
        biases = np.array([1.])
        weights = np.reshape(np.arange(9),(1,1,3,3))
        network[0].biases = biases
        network[0].weights = weights

        output = dnn_homebrew.forward(network, img.flatten())
        true_sum = 3406.0
        my_sum = np.sum(output)
        self.assertEqual(my_sum, true_sum)

    def test_single_max_pool_layer(self):
        img=np.reshape(np.arange(0,32),(4,4,2))
        img=np.transpose(img, (2,0,1))
        network=[nets.MaxPoolLayer((4,4), (2,2), 2)]
        output = dnn_homebrew.forward(network, img.flatten())

        my_sum = np.sum(output)
        true_sum=164.0
        self.assertEqual(my_sum, true_sum)

    def test_single_dense_layer(self):
        # input size 4, output size 3
        img=np.arange(4)
        weights = np.reshape(np.arange(12),(4,3))
        biases = np.arange(3)
        network=[nets.DenseLayer(4, 3)]
        network[0].weights = weights
        network[0].biases=biases
        output = dnn_homebrew.forward(network, img.flatten())
        my_sum = np.sum(output)
        true_sum=147.0
        self.assertEqual(my_sum, true_sum)



if __name__ == '__main__':
    unittest.main()
