# various layers and activation functions

import numpy as np

def convolution_forward(data, filters, bias, stride = 1, padding = 0):
    dump1, data_height, data_width = data.shape
    filter_count, dump2, filter_height, filter_width = filters.shape
    if padding > 0:
        padded_data = np.pad(data, ((0, 0), (padding, padding), (padding, padding)), 'constant')
    else:
        padded_data = data

    output_height = int((data_height - filter_height + 2 * padding) / stride) + 1
    output_width = int((data_width - filter_width + 2 * padding) / stride) + 1
    output = np.zeros((filter_count, output_height, output_width))

    for i in range(filter_count):
        for j in range(output_height):
            for k in range(output_width):
                start_height = j * stride
                end_height = start_height + filter_height
                start_width = k * stride
                end_width = start_width + filter_width
                area = padded_data[:, start_height:end_height, start_width:end_width]

                output[i, j, k] = np.sum(area * filters[i]) + bias[i]
    
    return output

def relu_forward(x):
    return np.maximum(x, 0)

def relu_backward(diff, x):
    dx = diff.copy()
    dx[x <= 0] = 0

    return dx

def max_pool_forward(data, pool_size = 2, stride = 2):
    count, height, width = data.shape
    output_height = int((height - pool_size) / stride) + 1
    output_width = int((width - pool_size) / stride) + 1
    output = np.zeros((count, output_height, output_width))
    cache = {}

    for i in range(count):
        for j in range(output_height):
            for k in range(output_width):
                start_height = j * stride
                end_height = start_height + pool_size
                start_width = k * stride
                end_width = start_width + pool_size
                area = data[i, start_height:end_height, start_width:end_width]

                output[i, j, k] = np.max(area)
                cache[(i, j, k)] = (start_height, end_height, start_width, end_width, np.argmax(area))
    
    return output, cache

def max_pool_backward(diff, cache, data_shape):
    count, height, width = data_shape
    dx = np.zeros((count, height, width))

    for (i, j, k), (start_height, end_height, start_width, end_width, index) in cache.items():
        area = np.zeros((end_height - start_height, end_width - start_width))
        np.put(area, index, diff[i, j, k])
        dx[i, start_height:end_height, start_width:end_width] += area
    
    return dx

def fully_connected_forward(data, weights, bias):
    output = np.dot(weights, data) + bias

    return output

def fully_connected_backward(diff, data, weights):
    dw = np.outer(diff, data)
    db = diff
    dx = np.dot(weights.T, diff)

    return dx, dw, db

def softmax_forward(x):
    exps = np.exp(x - np.max(x))

    return exps / np.sum(exps)

def softmax_backward(prediction, target):
    return prediction - target

def cross_entropy_loss(prediction, target):
    return -np.sum(target * np.log(prediction + 1e-8))