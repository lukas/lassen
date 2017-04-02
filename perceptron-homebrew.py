import data, weights
import numpy as np

def log_softmax(w):
    assert len(w.shape) == 1
    max_weight = np.max(w, axis=0)
    rightHandSize = np.log(np.sum(np.exp(w - max_weight), axis=0))

    return w - (max_weight + rightHandSize)

def gradient(input, output_gradient):
    return np.multiply(input, output_gradient)


def loss(images, labels, weights, bias):
    print(images.shape)
    print(labels.shape)
    print(weights.shape)
    print(bias.shape)

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


        # # print(log_likelihoods)
        # print("%s %s" % (np.argmax(log_likelihoods), np.argmax(labels[i])))
        # # print(np.sum(np.exp(log_softmax(likelihoods[i]))))
        # # print

    print(sum_loss)
    print(sum_grad_weights)
    print(sum_grad_bias)

    return sum_loss, sum_grad_weights, sum_grad_bias


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

    (l, grad_weights, grad_bias) = loss(image_batch, label_batch, w, b)

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
