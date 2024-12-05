import numpy as np
from numpy import random
from numpy import loadtxt
import math

class MLP:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        self.inputSize = inputSize
        self.hiddenSizes = hiddenSizes
        self.outputSize = outputSize
        self.number_of_layers = len(hiddenSizes) + 1

        self.weights = []
        self.biases = []
        sizes = [inputSize] + hiddenSizes + [outputSize]
        for i in range(1, self.number_of_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i-1]))
            self.biases.append(np.random.randn(sizes[i], 1))

    def backward(self, X, y):
        m = X.shape[1]
        gradients = []
        dZ = self.activations[-1] - y
        for i in range(self.number_of_layers - 1, -1, -1):
            dW = (1 / m) * np.dot(dZ, self.activations[i].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))

            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)
                dZ = dA * self.gradient_tanh(self.z[i - 1])

        return gradients[::-1]

    def forward(self, X):
        self.activations = [X]
        self.z = []
        for i in range(self.number_of_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            if i < self.number_of_layers - 1:
                a = self.tanh(z)
            else:
                a = z
            self.activations.append(a)
        return self.activations[-1]

    def update_parameters(self, gradients, learning_rate):
        # Update parameters using gradients and learning rate
        for i in range(self.number_of_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def tanh(self, Z):
        return np.tanh(Z)

    def gradient_tanh(self, Z):
        return 1 - np.tanh(Z)**2