# main script to load data, train, and test the CNN model

import numpy as np
import pickle as pk

from tools import *
from cnn import *

def get_batches(x, y, batch_size):
    for i in range(0, x.shape[0], batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

def main():
    train_label_path = 'data/train_labels'
    train_image_path = 'data/train_images'
    test_label_path = 'data/test_labels'
    test_image_path = 'data/test_images'

    print("Loading data...")
    train_labels = load_labels(train_label_path)
    train_images = load_images(train_image_path)
    test_labels = load_labels(test_label_path)
    test_images = load_images(test_image_path)
    print("Data loaded successfully.")

    print("\nSubsampling data...")
    subsample = 60000
    train_labels = train_labels[:subsample]
    train_images = train_images[:subsample]
    print("Data subsampled.")

    print("\nEncoding labels...")
    train_labels = encode(train_labels)
    test_labels = encode(test_labels)
    print("Labels encoded.")

    print("\nPreprocessing images...")
    train_images = preprocess(train_images)
    test_images = preprocess(test_images)
    print("Images preprocessed.")

    cnn = CNN()
    epochs = 8
    learn_rate = 0.001
    batch_size = 256

    print("\nStarting training...")
    for epoch in range(epochs):
        permutation = np.random.permutation(train_images.shape[0])
        shuffled_x = train_images[permutation]
        shuffled_y = train_labels[permutation]

        total_batches = int(np.ceil(shuffled_x.shape[0] / batch_size))
        batch_num = 0

        for x_batch, y_batch in get_batches(shuffled_x, shuffled_y, batch_size):
            batch_num += 1
            
            for x, y in zip(x_batch, y_batch):
                output = cnn.forward(x)
                cnn.backward(y, learn_rate)

            progress = (batch_num / total_batches) * 100
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_num}/{total_batches}, Progress: {progress:.2f}%", end='\r')

        print(f"Epoch {epoch+1}/{epochs} completed      ")
    print("Training completed.")

    print("\nSaving model weights...")
    weights = {
        'conv1_filters': cnn.conv1_filters,
        'conv1_bias': cnn.conv1_bias,
        'conv2_filters': cnn.conv2_filters,
        'conv2_bias': cnn.conv2_bias,
        'fc_weights': cnn.fc_weights,
        'fc_bias': cnn.fc_bias
    }

    with open('cnn/weights.pkl', 'wb') as file:
        pk.dump(weights, file)
    print("Model weights saved to 'cnn/weights.pkl'.")

    print("\nStarting testing...")
    correct = 0
    total = len(test_images)
    test_num = 0

    for x, y_true in zip(test_images, test_labels):
        test_num += 1
        
        output = cnn.forward(x)
        prediction = np.argmax(output)
        true_label = np.argmax(y_true)

        if prediction == true_label:
            correct += 1

        progress = (test_num / total) * 100
        print(f"Testing: {progress:.2f}% completed.", end='\r')

    accuracy = (correct / total) * 100
    print(f"Testing completed.")
    
    print(f"\nAccuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()