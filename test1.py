from multiLayerPerceptron import MLP
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    arr = np.array([[0,0], [1,0], [0,1], [0,0]])
    arr2 = np.array([[0], [1], [1], [0]])
    average = (0,0,0,0)

    for tests in range(10):
        mlp=MLP(2, [3, 3],1)
        for epoch in range(500):
            outputs = mlp.forward(arr.T)
            gradients = mlp.backward(arr.T, arr2.T)
            mlp.update_parameters(gradients, 0.01)
            loss = np.mean((outputs - arr2.T)**2)

        test_output = mlp.forward(arr.T)
        average = average + test_output

    average=average/10
    courses = ('0 0', '1 0', '0 1', '1 1')
    values = list(list(average[0]))

    fig = plt.figure(figsize = (10, 5))
    plt.bar(courses, values, color='maroon')
    plt.xlabel("XOR Input")
    plt.ylabel("XOR Output")
    plt.title("XOR Neural Network Results")
    plt.show()
