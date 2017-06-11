class LSTMLayer(Layer):
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
        i = sigmoid(np.dot(w_i.T, x) + np.dot(u_i, h_old)+b_i)
        c_tilde = np.tanh(np.dot(w_c.T, x)+np.dot(u_c, h_old)+b_c)
        f = sigmoid(np.dot(w_f.T, x) + np.dot(u_f, h_old)+b_f)
        c_new = i * c_tilde + f * c_old
        o = sigmoid(np.dot(w_o.T, x) + np.dot(u_o, h_old)+b_o)
        h_new = o * np.tanh(c_new)

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
