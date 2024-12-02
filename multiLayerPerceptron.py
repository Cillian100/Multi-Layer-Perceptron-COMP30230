import numpy as np
import random as rand

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        # Initialize the weights and biases for each layer
        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i-1]))
            self.biases.append(np.random.randn(sizes[i], 1))

    def backward(self, X, y):
        m = X.shape[1]
        #m = X.shape
        gradients = []
        dZ = self.activations[-1] - y
        for i in range(self.num_layers - 1, -1, -1):
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
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            if i < self.num_layers - 1:
                a = self.tanh(z)
            else:
                a = z
            self.activations.append(a)
        return self.activations[-1]

    def update_parameters(self, gradients, learning_rate):
        # Update parameters using gradients and learning rate
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def tanh(self, Z):
        return np.tanh(Z)

    def gradient_tanh(self, Z):
        return 1 - np.tanh(Z)**2


if __name__ == "__main__":
    val = rand.random()
    arr = [[0 for x in range(4)] for x in range(500)]
    if val>0.5:
        arr[0][0]=1
    else:
        arr[0][0]=-1


    print(arr)

if __name__ == "main":
    arr = np.array([[0,0], [1,0], [0,1], [0,0]])
    arr2 = np.array([[0], [1], [1], [0]])

    print(arr.T)
    mlp=MLP(2, [3, 3],1)

    for epoch in range(100):
        outputs = mlp.forward(arr.T)
        gradients = mlp.backward(arr.T, arr2.T)
        mlp.update_parameters(gradients, 0.01)
        loss = np.mean((outputs - arr2.T)**2)
        print(f"Epoch {epoch+1} - Loss: {loss}")

    test_output = mlp.forward(arr.T)
    test_loss = np.mean((test_output - arr2.T) ** 2)
    print(f"Test Loss: {test_loss}")
    print(test_output)