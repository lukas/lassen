import numpy as np

def load_mnist(data_filename, label_filename):
    images = []
    labels = []

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

    return images, labels
