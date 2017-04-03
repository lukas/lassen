import data, weights
import numpy as np
import sys

def log_softmax(w):
    assert len(w.shape) == 1
    max_weight = np.max(w, axis=0)
    rightHandSize = np.log(np.sum(np.exp(w - max_weight), axis=0))

    return w - (max_weight + rightHandSize)

def gradient(input, output_gradient):
    return np.multiply(input, output_gradient)


def loss(images, labels, weights, bias):
    likelihoods = np.dot(images, weights) + bias

    sum_grad_weights = np.zeros(weights.shape)
    sum_grad_bias = np.zeros(bias.shape)
    sum_loss = 0.0

    for i in range(len(labels)):
        # print(likelihoods[i])
        log_likelihoods = log_softmax(likelihoods[i])
        #print("LL", log_likelihoods)
        loss = -np.dot(labels[i], log_likelihoods)

        sum_loss += loss

        d_loss = np.exp(log_likelihoods) - labels[i]

        grad_bias = np.zeros(10)
        grad_weights = np.zeros(weights.shape)
        for j in range(10):
            grad_weights[:,j] = gradient(images[i], d_loss[j])
            grad_bias[j] = d_loss[j]

        sum_grad_weights += grad_weights
        sum_grad_bias += grad_bias

    return sum_loss, sum_grad_weights, sum_grad_bias

def accuracy(images, labels, w, b):
    likelihoods = np.dot(images, w) + b
    guess = np.argmax(likelihoods, axis=1)
    answer = np.argmax(labels, axis=1)
    return np.sum(np.equal(guess, answer))/len(guess)

def sgd(images, labels, w, b, test_images, test_labels):
    num_epochs = 100
    num_batches = 100
    learn_rate = 0.001
    batch_size = 100
    for _ in range(num_epochs):
        for _ in range(num_batches):
          rand_idx = np.floor(np.multiply(np.random.rand(batch_size), len(images))).astype(int)
          batch_labels = labels[rand_idx,:]
          batch_images = images[rand_idx,:]

          (l, grad_weights, grad_bias) = loss(batch_images, batch_labels, w, b)
          w -= grad_weights / batch_size * learn_rate
          b -= grad_bias / batch_size * learn_rate
          sys.stdout.write("Loss: %.3f  \r" % (l) )
          sys.stdout.flush()

        print("Train Accuracy %.2f%% " % (100*accuracy(images, labels, w, b)), end="")
        print("Test Accuracy  %.2f%% " % (100*accuracy(test_images, test_labels, w, b)))



def main():
    images, labels = data.load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
    test_images, test_labels = data.load_mnist("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte")
    images = images / 255.0
    test_images = test_images / 255.0

    tensorflow_weights = weights.load_weights_from_tensorflow("./tensorflow-checkpoint")
    tensorflow_bias = weights.load_bias_from_tensorflow("./tensorflow-checkpoint")

    w = np.zeros((28*28, 10))
    b = np.zeros(10)

    image_batch = images[[1], :]
    label_batch = labels[[1]]

    epsilon = 0.0001

    sgd(images, labels, w, b, test_images, test_labels)

    #(l2, grad_weights, grad_bias) = loss(image_batch, label_batch, w, b)


    # correct = 0
    # for (image, label) in zip(test_images, test_labels):
    #     activation = np.dot(image, tensorflow_weights) + tensorflow_bias
    #     print("%s %s" % (np.argmax(activation), np.argmax(label)))
    #     if np.argmax(activation) == np.argmax(label):
    #         correct += 1
    # print(float(correct) / len(images))

if __name__ == "__main__":
    main()
