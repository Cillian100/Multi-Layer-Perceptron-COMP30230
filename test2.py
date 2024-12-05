from matplotlib import pyplot as plt

from multiLayerPerceptron import MLP
from numpy import random
import math
import numpy as np
import pandas as pd

if __name__ == "__main__":
    x = random.choice([1, -1], size=(500, 4))
    arr = []
    for loop in range(500):
        arr.append(math.sin(x[loop][0] - x[loop][1] + x[loop][2] - x[loop][3]))

    vectorInput=np.array(x)
    vectorOutput=np.array(arr)
    mlp=MLP(4, [5,5], 1)

    for epoch in range(1000000):
        outputs = mlp.forward(vectorInput.T)
        gradients = mlp.backward(vectorInput.T, vectorOutput.T)
        mlp.update_parameters(gradients, 0.01)
        loss = np.mean((outputs-vectorOutput.T)**2)



    test_output = mlp.forward(vectorInput.T)
    x = pd.Series(test_output[0])
    y = pd.Series(arr)
    correlation = y.corr(x)
    print(correlation)
    plt.scatter(x, y)
    plt.plot(np.unique(y), np.poly1d(np.polyfit(x, y, 1)) (np.unique(y)), color='red' )
    print(y)
    plt.title(f"1000000 epochs {correlation}")
    plt.show()