#!/usr/bin/env python

import os
from struct import unpack
from array import array as array_unpack
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from shutil import rmtree
from subprocess import run
from mlp import MLP
from cnn import CNN
from my_lib import labels_to_one_hot, relu, cross_entropy, softmax, sigmoid, tqdm

MNIST_DATA_DIR = 'mnist_data'
TEST_IMAGES_PATH = os.path.join(MNIST_DATA_DIR, 't10k-images.idx3-ubyte')
TEST_LABELS_PATH = os.path.join(MNIST_DATA_DIR, 't10k-labels.idx1-ubyte')
TRAIN_IMAGES_PATH = os.path.join(MNIST_DATA_DIR, 'train-images.idx3-ubyte')
TRAIN_LABELS_PATH = os.path.join(MNIST_DATA_DIR, 'train-labels.idx1-ubyte')

if not os.path.isdir(MNIST_DATA_DIR):
    print('Downloading MNIST data...')
    os.mkdir(MNIST_DATA_DIR)
    DOWNLOAD_PATH = 'mnist-dataset.zip'
    urlretrieve('https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset', DOWNLOAD_PATH)
    run(['unzip', DOWNLOAD_PATH, '-d', MNIST_DATA_DIR])
    os.remove(DOWNLOAD_PATH)
    for filename in os.listdir(MNIST_DATA_DIR):
        path = os.path.join(MNIST_DATA_DIR, filename)
        if os.path.isdir(path):
            rmtree(path)
    print('Download complete.')


def read_binary(path, magic, head_size):
    with open(path, 'rb') as file:
        magic_got, *head = unpack('>' + 'I' * (head_size + 1), file.read(4 * (head_size + 1)))
        if magic_got != magic:
            raise ValueError(f'Magic number mismatch, expected {magic}, got {magic_got}')
        data = array_unpack('B', file.read())
    return *head, data


def read_images_labels(images_filepath, labels_filepath):
    size, labels = read_binary(labels_filepath, 2049, 1)
    size, rows, cols, image_data = read_binary(images_filepath, 2051, 3)
    return np.array(image_data).reshape(size, rows, cols), np.array(labels)


train_images, train_labels = read_images_labels(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)
test_images, test_labels = read_images_labels(TEST_IMAGES_PATH, TEST_LABELS_PATH)

np.random.seed(1108)
train_index = np.random.choice(len(train_images), 1024, replace=False)
train_images = train_images[train_index]
train_labels = labels_to_one_hot(train_labels[train_index], 10)


def display_confusion(model):
    confusion = np.zeros((10, 10), dtype=int)
    for image, label in tqdm(zip(test_images, test_labels), total=len(test_images)):
        confusion[label, model.predict(image).argmax()] += 1
    
    fig, ax = plt.subplots()
    im = ax.imshow(confusion)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    
    for i in range(10):
        for j in range(10):
            ax.text(j, i, confusion[i, j], ha='center', va='center', color='w')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()


if __name__ == '__main__':
    cnn = CNN(
        input_shape=train_images[0].shape,
        filter_size=6,
        num_filters=3,
        pool_size=2,
        hidden_layer_size=60,
        output_size=10
    )
    
    CNN_WEIGHTS_PATH = 'cnn_weights.npz'
    if os.path.isfile(CNN_WEIGHTS_PATH):
        cnn.load(CNN_WEIGHTS_PATH)
    else:
        losses = cnn.train(train_images, train_labels, [0.2] * 256)
        cnn.save(CNN_WEIGHTS_PATH)
        plt.plot(losses)
        plt.show()
        plt.savefig("plot_cnn_mnist.png")
    
    display_confusion(cnn)

    mlp = MLP(
        layer_sizes=[train_images[0].size, 700, 500, 10],
        activations=[relu, sigmoid, softmax],
        loss=cross_entropy
    )
    
    MLP_WEIGHTS_PATH = 'mlp_weights.npz'
    if os.path.isfile(MLP_WEIGHTS_PATH):
        mlp.load(MLP_WEIGHTS_PATH)
    else:
        losses = mlp.train(train_images, train_labels, [0.2] * 256, 64)
        mlp.save(MLP_WEIGHTS_PATH)
        plt.plot(losses)
        plt.show()
        plt.savefig("plot_mlp_mnist.png")
    
    display_confusion(mlp)