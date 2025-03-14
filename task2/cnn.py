import numpy as np
from itertools import count
from scipy.signal import correlate2d
from my_lib import relu, softmax, cross_entropy, sigmoid, tqdm


class Convolution:
    def __init__(self, input_shape, filter_size, num_filters, rng):
        input_height, input_width = input_shape
        self.filters = rng.standard_normal((num_filters, filter_size, filter_size))
        self.filters /= filter_size * np.sqrt(num_filters / 2)
        self.biases = rng.standard_normal((num_filters, input_height - filter_size + 1, input_width - filter_size + 1))

    def forward(self, input_data):
        self.input_data = input_data
        self.score = np.array([correlate2d(input_data, f, mode='valid') for f in self.filters])
        self.score += self.biases
        return relu(self.score)

    def backward(self, dloss_dout, learning_rate):
        dloss_dout *= self.score > 0
        dloss_dinput = np.zeros_like(self.input_data, dtype=float)
        dloss_dfilters = np.zeros_like(self.filters)

        for i in range(self.filters.shape[0]):
            dloss_dfilters[i] = correlate2d(self.input_data, dloss_dout[i], mode='valid')
            dloss_dinput += correlate2d(dloss_dout[i], self.filters[i], mode='full')
        
        self.filters -= learning_rate * dloss_dfilters
        self.biases -= learning_rate * dloss_dout
        return dloss_dinput

    def save(self, data):
        data.append(self.filters)
        data.append(self.biases)

    def load(self, data):
        self.filters, self.biases, *rest = data
        return rest


class MaxPooling:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):
        self.input_data = input_data
        return np.array([
            [
                [
                    np.max(input_data[n, i:i + self.pool_size, j:j + self.pool_size])
                    for j in range(0, input_data.shape[2], self.pool_size)
                ] for i in range(0, input_data.shape[1], self.pool_size)
            ] for n in range(input_data.shape[0])
        ])

    def backward(self, dloss_dout, learning_rate):
        dloss_dinput = np.zeros_like(self.input_data)
        for n in range(dloss_dinput.shape[0]):
            for i in range(0, dloss_dinput.shape[1], self.pool_size):
                for j in range(0, dloss_dinput.shape[2], self.pool_size):
                    window = self.input_data[n, i:i + self.pool_size, j:j + self.pool_size]
                    max_index = np.argmax(window)
                    max_i, max_j = np.unravel_index(max_index, window.shape)
                    dloss_dinput[n, i + max_i, j + max_j] = dloss_dout[n, i // self.pool_size, j // self.pool_size]
        return dloss_dinput

    def save(self, data):
        pass

    def load(self, data):
        return data


class FullyConnected:
    def __init__(self, input_size, output_size, activation, rng):
        self.activation = activation
        self.weights = rng.standard_normal((output_size, input_size)) / 100
        self.biases = np.zeros(output_size)

    def forward(self, input_data):
        self.input_data = input_data
        self.score = self.weights @ input_data.flatten() + self.biases
        return self.activation(self.score)

    def backward(self, dloss_dout, learning_rate):
        dloss_dscore = np.dot(self.activation.derivative(self.score), dloss_dout)
        dloss_dweights = np.outer(dloss_dscore, self.input_data.flatten())
        dloss_dbiases = dloss_dscore
        dloss_dinput = np.dot(self.weights.T, dloss_dscore).reshape(self.input_data.shape)
        
        self.weights -= learning_rate * dloss_dweights
        self.biases -= learning_rate * dloss_dbiases
        return dloss_dinput

    def save(self, data):
        data.append(self.weights)
        data.append(self.biases)

    def load(self, data):
        self.weights, self.biases, *rest = data
        return rest


class CNN:
    def __init__(self, input_shape, filter_size, num_filters, pool_size, hidden_layer_size, output_size, seed=1108):
        self.rng = np.random.default_rng(seed)
        self.convolution = Convolution(input_shape, filter_size, num_filters, self.rng)
        self.pooling = MaxPooling(pool_size)
        pooled_shape = (
            num_filters,
            -((input_shape[0] - filter_size + 1) // -pool_size),
            -((input_shape[1] - filter_size + 1) // -pool_size)
        )
        self.fully_connected1 = FullyConnected(
            np.prod(pooled_shape),
            hidden_layer_size,
            sigmoid,
            self.rng
        )
        self.fully_connected2 = FullyConnected(
            hidden_layer_size,
            output_size,
            softmax,
            self.rng
        )
        self.layers = [
            self.convolution,
            self.pooling,
            self.fully_connected1,
            self.fully_connected2
        ]

    def train(self, x, y, learning_rates):
        losses = []
        n_samples = len(x)
        with tqdm(total=len(learning_rates) * n_samples) as progress:
            for learning_rate in learning_rates:
                loss = 0
                for image, label in zip(x, y):
                    output = image
                    for layer in self.layers:
                        output = layer.forward(output)
                    loss += cross_entropy(label, output) / n_samples
                    dloss_dout = cross_entropy.derivative(label, output) / n_samples
                    for layer in reversed(self.layers):
                        dloss_dout = layer.backward(dloss_dout, learning_rate)
                    progress.update()
                losses.append(loss)
        return losses

    def predict(self, image):
        output = image
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def save(self, path):
        data = []
        for layer in self.layers:
            layer.save(data)
        np.savez(path, *data)

    def load(self, path):
        data_dict = np.load(path)
        data = [data_dict[f'arr_{i}'] for i in count() if f'arr_{i}' in data_dict]
        for layer in self.layers:
            data = layer.load(data)
