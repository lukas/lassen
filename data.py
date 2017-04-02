import numpy as np

def load_mnist(data_filename, label_filename):
    images = []
    labels = []
    num_classes = 10

    with open(data_filename, "rb") as f:
        magic = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        col_count = int.from_bytes(f.read(4), 'big')
        images = np.fromfile(f, dtype=np.dtype(np.uint8))
        images = images.reshape(image_count, row_count*col_count)

    with open(label_filename, "rb") as f:
        magic = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        labels = np.fromfile(f, dtype=np.uint8)

    labels = convert_to_one_hot(labels, num_classes)

    return images, labels


def convert_to_one_hot(array, num_classes):
    num_labels = len(array)
    one_hot_labels = np.zeros((num_labels, num_classes))
    one_hot_labels[np.arange(num_labels), array] = 1
    return one_hot_labels
