import numpy as np


class QuadraticCost(object):

    def fn(self, a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    def delta(self, z, a, y):
        return (a - y) * self.sigmoid_prime(z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
