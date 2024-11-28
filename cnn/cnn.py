# class to implement the CNN model

import numpy as np

from layer import *

class CNN:
    def __init__(self):
        self.num_classes = 10

        self.conv1_filters = np.random.randn(8, 1, 3, 3) * np.sqrt(1. / 9)
        self.conv1_bias = np.zeros(8)

        self.conv2_filters = np.random.randn(16, 8, 3, 3) * np.sqrt(1. / 72)
        self.conv2_bias = np.zeros(16)

        self.flat_size = 16 * 7 * 7

        self.fc_weights = np.random.randn(self.num_classes, self.flat_size) * np.sqrt(1. / self.flat_size)
        self.fc_bias = np.zeros(self.num_classes)
    
    def forward(self, x):
        self.x = x

        self.conv1 = convolution_forward(x, self.conv1_filters, self.conv1_bias, padding = 1)
        self.relu1 = relu_forward(self.conv1)
        self.pool1, self.pool1_cache = max_pool_forward(self.relu1)

        self.conv2 = convolution_forward(self.pool1, self.conv2_filters, self.conv2_bias, padding = 1)
        self.relu2 = relu_forward(self.conv2)
        self.pool2, self.pool2_cache = max_pool_forward(self.relu2)

        self.flat = self.pool2.flatten()

        self.fc = fully_connected_forward(self.flat, self.fc_weights, self.fc_bias)

        self.output = softmax_forward(self.fc)

        return self.output
    
    def backward(self, y_true, learn_rate = 0.001):
        dloss = softmax_backward(self.output, y_true)

        dflat, dfc_weights, dfc_bias = fully_connected_backward(dloss, self.flat, self.fc_weights)

        self.fc_weights -= learn_rate * dfc_weights
        self.fc_bias -= learn_rate * dfc_bias

        dpool2 = dflat.reshape(self.pool2.shape)

        drelu2 = max_pool_backward(dpool2, self.pool2_cache, self.relu2.shape)

        dconv2 = relu_backward(drelu2, self.conv2)

        dconv2_filters, dconv2_bias, dpool1 = self.convolution_backward(dconv2, self.conv2_filters, self.pool1, padding = 1)

        self.conv2_filters -= learn_rate * dconv2_filters
        self.conv2_bias -= learn_rate * dconv2_bias

        drelu1 = max_pool_backward(dpool1, self.pool1_cache, self.relu1.shape)

        dconv1 = relu_backward(drelu1, self.conv1)

        dconv1_filters, dconv1_bias, dump = self.convolution_backward(dconv1, self.conv1_filters, self.x, padding = 1)

        self.conv1_filters -= learn_rate * dconv1_filters
        self.conv1_bias -= learn_rate * dconv1_bias
    
    def convolution_backward(self, diff, filters, data, padding = 0, stride = 1):
        filter_count, dump1, filter_height, filter_width = filters.shape
        data_count, data_height, data_width = data.shape
        dump2, output_height, output_width = diff.shape

        dfilters = np.zeros_like(filters)
        dbias = np.zeros(filter_count)
        ddata = np.zeros_like(data)
        if padding > 0:
            padded_data = np.pad(data, ((0, 0), (padding, padding), (padding, padding)), 'constant')
            dpadded_data = np.pad(ddata, ((0, 0), (padding, padding), (padding, padding)), 'constant')
        else:
            padded_data = data
            dpadded_data = ddata
        
        for i in range(filter_count):
            dbias[i] += np.sum(diff[i])

            for j in range(data_count):
                for k in range(output_height):
                    for l in range(output_width):
                        start_height = k * stride
                        end_height = start_height + filter_height
                        start_width = l * stride
                        end_width = start_width + filter_width
                        area = padded_data[j, start_height:end_height, start_width:end_width]

                        dfilters[i, j] += area * diff[i, k, l]
                        dpadded_data[j, start_height:end_height, start_width:end_width] += filters[i, j] * diff[i, k, l]
        
        if padding > 0:
            ddata = dpadded_data[:, padding:-padding, padding:-padding]
        else:
            ddata = dpadded_data
        
        return dfilters, dbias, ddata