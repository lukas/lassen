import data, weights
import numpy as np

def log_softmax(w):
    assert len(w.shape) == 1
    max_weight = np.max(w, axis=0)
    rightHandSize = np.log(np.sum(np.exp(w - max_weight), axis=0))
    return w - (max_weight + rightHandSize)

def loss(images, labels, weights, bias):
    print(images.shape)
    print(labels.shape)
    print(weights.shape)
    print(bias.shape)

    likelihoods = np.dot(images, weights) + bias


    for i in range(1):
        # print(likelihoods[i])
        log_likelihoods = log_softmax(likelihoods[i])
        loss = -np.dot(labels[i], log_likelihoods)
        d_loss = labels[i] - likelihoods[i]
        print(d_loss)

        for j in range(len(log_likelihoods)):
            epsilon = 1.0e-5

            new_likelihoods = likelihoods[i].copy()
            new_likelihoods[j] += epsilon
            new_log_likelihoods = log_softmax(new_likelihoods)
            new_loss = -np.dot(labels[i], new_log_likelihoods)

            print("%i : %f %f" %
                (j,
                new_loss - loss,
                (new_loss - loss) / epsilon),
                )


        # # print(log_likelihoods)
        # print("%s %s" % (np.argmax(log_likelihoods), np.argmax(labels[i])))
        # # print(np.sum(np.exp(log_softmax(likelihoods[i]))))
        # # print

def main():
    images, labels = data.load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
    test_images, test_labels = data.load_mnist("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte")
    images = images / 255.0
    test_images = test_images / 255.0

    tensorflow_weights = weights.load_weights_from_tensorflow("./tensorflow-checkpoint")
    tensorflow_bias = weights.load_bias_from_tensorflow("./tensorflow-checkpoint")

    loss(images, labels, tensorflow_weights, tensorflow_bias)

    # correct = 0
    # for (image, label) in zip(test_images, test_labels):
    #     activation = np.dot(image, tensorflow_weights) + tensorflow_bias
    #     print("%s %s" % (np.argmax(activation), np.argmax(label)))
    #     if np.argmax(activation) == np.argmax(label):
    #         correct += 1
    # print(float(correct) / len(images))

if __name__ == "__main__":
    main()
