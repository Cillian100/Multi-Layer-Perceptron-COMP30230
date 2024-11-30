import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def backward(self, X, y, output, learning_rate):
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (
                    1 - self.hidden_output)

        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)
        self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if(epoch+1)%100 == 0:
                loss = -np.sum(y * np.log(output)) / X.shape[0]
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')


    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)



if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_mean = np.mean(X_train)
    X_train_std = np.std(X_train)
    X_train_std = (X_train - X_train_mean)/X_train_std
    X_train = (X_test - X_train_mean)/X_train_std

    y_train = y_train.reshape(-1,1)
    y_test = y_test(-1,1)

    input_size = X_train.shape[1]
    hidden_sizes = [10, 10]
    output_size = y_train.shape[1]
    mlp=MLP(input_size, hidden_sizes, output_size)

    num_epoch=1000
    learning_rate=0.01

    for epoch in range(num_epoch):
        outputs = mlp.foward(X_train.T)
        gradients = mlp.backward(X_train.T, y_train.T)
        mlp.update_parameters(gradients, learning_rate)
