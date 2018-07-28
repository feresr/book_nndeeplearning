import random
import numpy as np


class NeuralNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for epoch in range(epochs):
            print("starting epoch:", epoch)
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(epoch))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = np.add(nabla_b, delta_nabla_b)
            nabla_w = np.add(nabla_w, delta_nabla_w)
        self.weights = [w - (nw / len(mini_batch)) * eta for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (nb / len(mini_batch)) * eta for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.sigmoid_prime(zs[-l])
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            nabla_b[-l] = delta

        return nabla_b, nabla_w

    def cost_derivative(self, output_activation, y):
        return output_activation - y
