# helper functions to process data

import numpy as np
import struct as st

def load_labels(path):
    with open(path, 'rb') as file:
        dump1, dump2 = st.unpack('>II', file.read(8))
        labels = np.fromfile(file, dtype = np.uint8)

    return labels

def load_images(path):
    with open(path, 'rb') as file:
        dump, size, rows, cols = st.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype = np.uint8).reshape(size, rows, cols)

    return images

def encode(labels):
    return np.eye(10)[labels]

def preprocess(images):
    images = images.astype(np.float32) / 255.0
    images = np.rot90(images, k = 1, axes = (1, 2))
    images = images[:, np.newaxis, :, :]

    return images